/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
/* Fixed-chart polish shares the weighted-cover pricing kernel. */
#if defined(HAVE_PTHREAD)
#include <pthread.h>
#endif
#include "polish_order.h"
#include <stdatomic.h>

struct IpcChartPolish {
    NsatCoverInput input;
    WeightedCover weighted;
    PricingBasis root;
    unsigned workers;
    bool ready;
};

typedef struct WarmWorker WarmWorker;
typedef struct {
#if defined(HAVE_PTHREAD)
    pthread_mutex_t mutex;
    pthread_cond_t changed;
#endif
    atomic_bool stop;
    atomic_uint idle;
    atomic_uint_fast64_t nodes;
    uint64_t node_limit, available, future, probe, checkpoint, extensions;
    uint64_t closed;
    unsigned started, waiting, active;
    int winner;
    NsatStopReason reason;
    WarmWorker *workers;
} WarmPool;

typedef enum { WORKER_IDLE, WORKER_ASSIGNED, WORKER_RUNNING } WorkerState;

struct WarmWorker {
#if defined(HAVE_PTHREAD)
    pthread_t thread; /* Accounted with the dynamically allocated worker. */
#endif
    WeightedCover weighted;
    Repricing pricing;
    CompactPricing compact;
    const IpcChartPolishQuery *query;
    WarmPool *pool;
    size_t *choices, *frequency, *selected;
    SearchFrame *frames;
    uint64_t *residuals;
    PolishCandidate *candidates;
    size_t levels, initial_count, depth, count;
    SavedBasis *task_basis;
    double *task_rhs;
    bool *task_surplus;
    size_t *task_coordinates;
    uint64_t *task_weights;
    uint64_t margin;
    int length;
    WorkerState state;
};

static uint64_t chart_polish_now(void)
{
    struct timespec current;
    if (clock_gettime(CLOCK_MONOTONIC, &current) != 0) return UINT64_MAX;
    return (uint64_t)current.tv_sec * UINT64_C(1000000000) + (uint64_t)current.tv_nsec;
}

static bool warm_polish_poll(void *context)
{
    WeightedCover *w = context;
    if (chart_polish_now() >= w->polish_deadline) return true;
    if (w->budget.stats.work < w->next_probe ||
        w->next_probe >= w->budget.limits.work) return false;
    if (w->progress == w->checkpoint_progress) {
        nsat_stop(&w->budget, NSAT_UNKNOWN, NSAT_STOP_WORK);
        return true;
    }
    w->checkpoint_progress = w->progress;
    w->next_probe += w->probe_work;
    return false;
}

void ipc_chart_polish_close(IpcChartPolish *polish)
{
    if (!polish) return;
    WeightedCover *w = &polish->weighted;
    nsat_free(&w->budget, w->weight_bytes, w->mask_words * 8 * sizeof(*w->weight_bytes));
    free_basis(&w->budget, polish->root);
    free(polish);
}

IpcChartPolish *ipc_chart_polish_open(const PIChartView *chart, size_t memory_bytes,
                                    uint64_t deadline_nanoseconds, unsigned workers)
{
    if (!chart || !chart->bits || chart->cov_bits != 64 ||
        chart->rows <= 0 || chart->cols <= 0 || chart->words != (chart->rows - 1) / 64 + 1 ||
        deadline_nanoseconds <= chart_polish_now()) return NULL;

    /* Root and projection use one dense matrix each. All saved projected
     * inverses are allocated only while the owning node or task is live. */
    size_t rows = (size_t)chart->rows, cells;
    if (!nsat_mul(rows, rows, &cells) || cells > memory_bytes / sizeof(double) / 3 ||
        sizeof(IpcChartPolish) > memory_bytes) return NULL;
    IpcChartPolish *state = calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->workers = workers < 1 ? 1 : workers;
#if !defined(HAVE_PTHREAD)
    state->workers = 1;
#endif
    state->input.positives = rows;
    WeightedCover *w = &state->weighted;
    w->polish_deadline = deadline_nanoseconds;
    w->budget.limits = (NsatLimits){
        .work = UINT64_C(16000000000), .memory_bytes = memory_bytes,
        .poll = warm_polish_poll, .context = w
    };
    w->budget.stats.memory_bytes = w->budget.stats.peak_memory_bytes = sizeof(*state);
    w->input = &state->input;
    w->mask_words = (size_t)chart->words;
    w->columns = (uint64_t *)chart->bits;
    w->result.maximal = (size_t)chart->cols;
    w->probe_work = w->next_probe = UINT64_C(4000000000);
    return state;
}

static bool chart_polish_root(IpcChartPolish *state)
{
    WeightedCover *w = &state->weighted;
    NsatBudget *budget = &w->budget;
    w->weight_bytes = nsat_alloc(budget, w->mask_words * 8, sizeof(*w->weight_bytes));
    size_t count = w->mask_words * 64;
    uint64_t *weights = nsat_alloc(budget, count, sizeof(uint64_t));
    if (!w->weight_bytes || !weights ||
        !nsat_copy(budget, weights, NULL, count * sizeof(uint64_t)) ||
        !price_weights(w, weights, &state->root, false)) goto done;
    for (size_t byte = 0; byte < w->mask_words * 8; ++byte) {
        if (!nsat_work(budget, 256)) goto done;
        w->weight_bytes[byte][0] = 0;
        for (unsigned bits = 1; bits < 256; ++bits) {
            w->weight_bytes[byte][bits] = w->weight_bytes[byte][bits & (bits - 1)] +
                weights[byte * 8 + (unsigned)__builtin_ctz(bits)];
        }
    }
    state->ready = true;
done:
    nsat_free(budget, weights, count * sizeof(uint64_t));
    return state->ready;
}

/* QCA's reservations let each worker charge ordinary native operations
 * locally. The coordinator extends a shared probe only after a subtree
 * finishes. All workers retain their search state while waiting. */
#if defined(HAVE_PTHREAD)
static bool warm_worker_poll(void *context)
{
    WarmWorker *worker = context;
    WarmPool *pool = worker->pool;
    NsatBudget *budget = &worker->weighted.budget;
    if (atomic_load(&pool->stop)) return true;
    if (budget->limits.work - budget->stats.work > worker->margin) return false;

    pthread_mutex_lock(&pool->mutex);
    while (!pool->available && !atomic_load(&pool->stop)) {
        ++pool->waiting;
        pthread_cond_broadcast(&pool->changed);
        pthread_cond_wait(&pool->changed, &pool->mutex);
        --pool->waiting;
    }
    if (!atomic_load(&pool->stop)) {
        uint64_t grant = UINT64_C(1048576);
        if (grant < 2 * worker->margin) grant = 2 * worker->margin;
        if (grant > pool->available) grant = pool->available;
        pool->available -= grant;
        budget->limits.work += grant;
    }
    bool stopped = atomic_load(&pool->stop);
    pthread_mutex_unlock(&pool->mutex);
    return stopped;
}

#endif

static bool warm_node_claim(WarmWorker *worker)
{
    if (!worker->pool)
        return worker->weighted.result.nodes < (uint64_t)worker->query->node_limit;
    WarmPool *pool = worker->pool;
    if (atomic_load(&pool->stop)) return false;
    uint_fast64_t count = atomic_load_explicit(&pool->nodes, memory_order_relaxed);
    while (count < pool->node_limit) {
        if (atomic_compare_exchange_weak_explicit(&pool->nodes, &count, count + 1,
                memory_order_relaxed, memory_order_relaxed)) return true;
    }
    return false;
}

static NsatStatus warm_polish_visit(WarmWorker *worker, size_t depth, size_t parent_count);

/* An idle worker receives the exact current child, including the preceding
 * sibling exclusions and its parent's pricing state. The donor can unwind
 * as soon as the copy is complete. No worker replays a search prefix. */
static bool warm_offer(WarmWorker *donor, size_t depth, size_t count)
{
#if defined(HAVE_PTHREAD)
    WarmPool *pool = donor->pool;
    if (!pool || !atomic_load_explicit(&pool->idle, memory_order_relaxed)) return false;
    WeightedCover *w = &donor->weighted;
    const size_t rows = w->input->positives, cols = w->result.maximal;
    const SearchFrame *source = &donor->frames[depth - 1];
    const bool private_basis = donor->pricing.pivots &&
        source->basis.inverse != donor->pricing.root.inverse;
    size_t basis_rows = private_basis ? source->basis.rows : 0;
    if (!nsat_work(&w->budget, count + depth + w->mask_words + rows +
                   basis_rows * (basis_rows + 3))) return false;

    pthread_mutex_lock(&pool->mutex);
    bool offered = false;
    for (unsigned i = 0; i < pool->started && !atomic_load(&pool->stop); ++i) {
        WarmWorker *receiver = &pool->workers[i];
        if (receiver->state != WORKER_IDLE) continue;
        SavedBasis *saved = private_basis ? reserve_basis(donor->compact.live, basis_rows) : NULL;
        if (private_basis && !saved) break;
        SearchFrame *frame = &receiver->frames[depth - 1];
        *frame = *source;
        if (saved) {
            copy_basis_matrix(saved, &frame->basis);
            memcpy(receiver->task_rhs, source->basis.rhs, basis_rows * sizeof(double));
            memcpy(receiver->task_coordinates, source->basis.coordinates, basis_rows * sizeof(size_t));
            memcpy(receiver->task_surplus, source->basis.surplus, basis_rows * sizeof(bool));
            frame->basis.rhs = receiver->task_rhs;
            frame->basis.coordinates = receiver->task_coordinates;
            frame->basis.surplus = receiver->task_surplus;
        }
        if (source->weights) {
            /* Only residual rows have initialized local weights. */
            const uint64_t *remain = donor->residuals + depth * w->mask_words;
            for (size_t row = next_row(remain, w->mask_words, 0); row != SIZE_MAX;
                 row = next_row(remain, w->mask_words, row + 1))
                receiver->task_weights[row] = source->weights[row];
            frame->weights = receiver->task_weights;
        }
        memcpy(receiver->choices + (depth - 1) * cols,
               donor->choices + (depth - 1) * cols, count * sizeof(size_t));
        memcpy(receiver->residuals + depth * w->mask_words,
               donor->residuals + depth * w->mask_words, w->mask_words * sizeof(uint64_t));
        memcpy(receiver->selected, donor->selected, depth * sizeof(size_t));
        receiver->depth = depth;
        receiver->count = count;
        receiver->task_basis = saved;
        receiver->state = WORKER_ASSIGNED;
        atomic_fetch_sub_explicit(&pool->idle, 1, memory_order_relaxed);
        ++pool->active;
        offered = true;
        pthread_cond_broadcast(&pool->changed);
        break;
    }
    pthread_mutex_unlock(&pool->mutex);
    return offered;
#else
    (void)donor;
    (void)depth;
    (void)count;
    return false;
#endif
}

/* IPC still supplies the target, reduced-cost core, forced columns and
 * candidate order. The per-node integer bounds partition the remaining
 * candidates by excluding earlier siblings. */
static NsatStatus warm_polish_node(WarmWorker *worker, size_t depth, size_t parent_count)
{
    WeightedCover *w = &worker->weighted;
    const IpcChartPolishQuery *query = worker->query;
    const size_t cols = w->result.maximal, words = w->mask_words;
    uint64_t *rows = worker->residuals + depth * words;
    size_t *active = worker->choices + depth * cols;
    const size_t *parent = depth ? active - cols : active;
    SearchFrame *frame = worker->frames + depth;
    if (depth >= worker->levels) return NSAT_UNKNOWN;
    if (!warm_node_claim(worker)) return NSAT_UNKNOWN;
    NsatStatus status = prepare_search(w, rows, query->target - query->forced_len - (int)depth,
        parent, parent_count, active, frame, worker->frequency, worker->selected,
        &worker->pricing, depth);
    if (status == NSAT_SAT) worker->length = (int)depth + 1;
    if (status != NSAT_OK) return status;

    PolishCandidate *candidates = worker->candidates + depth * (size_t)query->max_row_degree;
    int count = 0;
    for (size_t i = 0; i < frame->count; ++i) {
        if (!nsat_work(&w->budget, 1)) return w->budget.status;
        size_t index = active[i];
        const uint64_t *mask = column(w, index);
        if (!contains_row(mask, frame->anchor)) continue;
        int added = 0;
        for (size_t word = 0; word < words; ++word)
            added += __builtin_popcountll(mask[word] & rows[word]);
        candidates[count++] = (PolishCandidate){
            .col = (int)index,
            .new_cover = added,
            .cover_count = query->coverage_counts[index],
            .lagr_score = query->reduced_costs ? query->reduced_costs[index] : 0.0,
            .weight = query->tie_weights ? query->tie_weights[index] : 0.0
        };
    }
    polish_sort_candidates(candidates, count);
    for (int i = 0; i < count; ++i) {
        size_t index = (size_t)candidates[i].col;
        size_t pos = 0;
        while (pos < frame->count && active[pos] != index) ++pos;
        if (!nsat_work(&w->budget, pos + 1)) return w->budget.status;
        if (pos == frame->count) return NSAT_ERROR;
        if (dominated(w, rows, active, frame->count, pos)) {
            continue;
        }
        if (w->budget.status != NSAT_OK) return w->budget.status;
        const uint64_t *mask = column(w, index);
        uint64_t *next = rows + words;
        for (size_t word = 0; word < words; ++word) next[word] = rows[word] & ~mask[word];
        worker->selected[depth] = index;
        size_t following = frame->count - pos - 1;
        if (!nsat_work(&w->budget, following + words)) return w->budget.status;
        memmove(active + pos, active + pos + 1, following * sizeof(*active));
        --frame->count;
        if (i + 1 < count && depth + 2 < (size_t)(query->target - query->forced_len) &&
            warm_offer(worker, depth + 1, frame->count)) continue;
        if (w->budget.status != NSAT_OK) return w->budget.status;
        status = warm_polish_visit(worker, depth + 1, frame->count);
        if (status != NSAT_UNSAT) return status;
        if (!worker->pool && !depth) ++w->progress;
    }
    return NSAT_UNSAT;
}

static NsatStatus warm_polish_visit(WarmWorker *worker, size_t depth, size_t parent_count)
{
    SavedBasis *mark = worker->compact.top;
    NsatStatus status = warm_polish_node(worker, depth, parent_count);
    restore_bases(&worker->compact, mark);
    return status;
}

static bool warm_worker_prepare(WeightedCover *w, WarmWorker *worker,
                                const PricingBasis *root, LiveBases *live)
{
    const size_t levels = worker->levels, rows = w->input->positives;
    const size_t cols = w->result.maximal, degree = (size_t)worker->query->max_row_degree;
    NsatBudget *budget = &w->budget;
    worker->weighted = *w;
    worker->weighted.result = (IpcWeightedCoverResult){.maximal = cols};
    worker->margin = rows * rows + 3 * rows + cols + 8192;
    worker->choices = allocate_matrix(budget, levels, cols, sizeof(size_t));
    worker->frequency = nsat_alloc(budget, rows, sizeof(size_t));
    worker->selected = nsat_alloc(budget, levels, sizeof(size_t));
    worker->frames = nsat_alloc(budget, levels, sizeof(SearchFrame));
    worker->residuals = allocate_matrix(budget, levels, w->mask_words, sizeof(uint64_t));
    worker->candidates = allocate_matrix(budget, levels, degree, sizeof(PolishCandidate));
    worker->pricing = (Repricing){
        .min_branches = 4, .pivots = 256,
        .root = *root, .compact = &worker->compact
    };
    if (!worker->choices || !worker->frequency || !worker->selected ||
        !worker->frames || !worker->residuals || !worker->candidates) return false;

    Repricing *p = &worker->pricing;
    const size_t cells = (levels - 1) * rows;
    p->rhs = nsat_alloc(budget, cells, sizeof(double));
    p->surplus = nsat_alloc(budget, cells, sizeof(bool));
    p->coordinates = nsat_alloc(budget, cells, sizeof(size_t));
    p->weights = nsat_alloc(budget, cells, sizeof(uint64_t));
    p->positions = nsat_alloc(budget, rows, sizeof(size_t));
    CompactPricing *c = &worker->compact;
    c->projection = allocate_matrix(budget, rows, rows, sizeof(double));
    c->prices = nsat_alloc(budget, rows, 2 * sizeof(double));
    c->live = live;
    worker->task_rhs = nsat_alloc(budget, rows, sizeof(double));
    worker->task_surplus = nsat_alloc(budget, rows, sizeof(bool));
    worker->task_coordinates = nsat_alloc(budget, rows, sizeof(size_t));
    worker->task_weights = nsat_alloc(budget, rows, sizeof(uint64_t));
    return p->rhs && p->surplus && p->coordinates && p->weights && p->positions &&
           c->projection && c->prices && worker->task_rhs && worker->task_surplus &&
           worker->task_coordinates && worker->task_weights;
}

static void warm_worker_release(WeightedCover *w, WarmWorker *worker)
{
    const size_t levels = worker->levels, rows = w->input->positives, cols = w->result.maximal;
    NsatBudget *budget = &w->budget;
    Repricing *p = &worker->pricing;
    CompactPricing *c = &worker->compact;
    const size_t cells = (levels - 1) * rows;
    release_basis(c->live, worker->task_basis);
    nsat_free(budget, worker->task_weights, rows * sizeof(uint64_t));
    nsat_free(budget, worker->task_coordinates, rows * sizeof(size_t));
    nsat_free(budget, worker->task_surplus, rows * sizeof(bool));
    nsat_free(budget, worker->task_rhs, rows * sizeof(double));
    nsat_free(budget, c->prices, rows * 2 * sizeof(double));
    nsat_free(budget, c->projection, rows * rows * sizeof(double));
    nsat_free(budget, p->positions, rows * sizeof(size_t));
    nsat_free(budget, p->weights, cells * sizeof(uint64_t));
    nsat_free(budget, p->coordinates, cells * sizeof(size_t));
    nsat_free(budget, p->surplus, cells * sizeof(bool));
    nsat_free(budget, p->rhs, cells * sizeof(double));
    nsat_free(budget, worker->candidates,
              levels * (size_t)worker->query->max_row_degree * sizeof(PolishCandidate));
    nsat_free(budget, worker->residuals, levels * w->mask_words * sizeof(uint64_t));
    nsat_free(budget, worker->frames, levels * sizeof(SearchFrame));
    nsat_free(budget, worker->selected, levels * sizeof(size_t));
    nsat_free(budget, worker->frequency, rows * sizeof(size_t));
    nsat_free(budget, worker->choices, levels * cols * sizeof(size_t));
}

static size_t warm_worker_fixed_bytes(const WeightedCover *w,
                                      const IpcChartPolishQuery *query, size_t levels)
{
    const size_t rows = w->input->positives, cols = w->result.maximal;
    return sizeof(WarmWorker) + levels * (
        cols * sizeof(size_t) + sizeof(size_t) + sizeof(SearchFrame) +
        w->mask_words * sizeof(uint64_t) +
        (size_t)query->max_row_degree * sizeof(PolishCandidate)) + rows * sizeof(size_t) +
        (levels - 1) * rows * (sizeof(double) + sizeof(bool) + sizeof(size_t) + sizeof(uint64_t)) +
        rows * (2 * sizeof(size_t) + (rows + 3) * sizeof(double) + sizeof(bool) + sizeof(uint64_t));
}

#if defined(HAVE_PTHREAD)
static void *warm_worker_run(void *context)
{
    WarmWorker *worker = context;
    WarmPool *pool = worker->pool;
    NsatBudget *budget = &worker->weighted.budget;
    pthread_mutex_lock(&pool->mutex);
    for (;;) {
        while (worker->state == WORKER_IDLE && !atomic_load(&pool->stop))
            pthread_cond_wait(&pool->changed, &pool->mutex);
        if (atomic_load(&pool->stop)) break;
        worker->state = WORKER_RUNNING;
        pthread_mutex_unlock(&pool->mutex);
        NsatStatus status = nsat_poll(budget)
            ? warm_polish_visit(worker, worker->depth, worker->count) : budget->status;
        release_basis(worker->compact.live, worker->task_basis);
        worker->task_basis = NULL;
        pthread_mutex_lock(&pool->mutex);
        pool->available += budget->limits.work - budget->stats.work;
        budget->limits.work = budget->stats.work;
        --pool->active;
        if (status == NSAT_SAT) {
            if (pool->winner < 0) pool->winner = (int)(worker - pool->workers);
            atomic_store(&pool->stop, true);
        } else if (status != NSAT_UNSAT && !atomic_load(&pool->stop)) {
            pool->reason = budget->stats.stop;
            if (!pool->reason && atomic_load(&pool->nodes) >= pool->node_limit)
                pool->reason = NSAT_STOP_WORK;
            if (!pool->reason) pool->reason = NSAT_STOP_STATE;
            atomic_store(&pool->stop, true);
        } else if (status == NSAT_UNSAT) {
            ++pool->closed;
            if (!pool->active) atomic_store(&pool->stop, true);
        }
        worker->state = WORKER_IDLE;
        atomic_fetch_add_explicit(&pool->idle, 1, memory_order_relaxed);
        pthread_cond_broadcast(&pool->changed);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static unsigned warm_run_pool(WarmPool *pool, WarmWorker *workers, unsigned count,
                              WeightedCover *w, uint64_t node_limit)
{
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) return 0;
    if (pthread_cond_init(&pool->changed, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        return 0;
    }
    atomic_init(&pool->stop, false);
    atomic_init(&pool->idle, 0);
    atomic_init(&pool->nodes, 0);
    pool->winner = -1;
    pool->node_limit = node_limit;
    pool->workers = workers;
    uint64_t hard = w->budget.limits.work;
    pool->probe = w->probe_work;
    uint64_t current = w->budget.stats.work;
    uint64_t probe_end = w->next_probe < hard ? w->next_probe : hard;
    pool->available = probe_end > current ? probe_end - current : 0;
    pool->future = hard - current - pool->available;
    if (w->progress > w->checkpoint_progress) pool->checkpoint = UINT64_MAX;
    pthread_mutex_lock(&pool->mutex);
    for (unsigned i = 0; i < count; ++i) {
        workers[i].pool = pool;
        workers[i].weighted.budget = (NsatBudget){.limits = {
            .poll = warm_worker_poll, .context = &workers[i]
        }};
        if (pthread_create(&workers[i].thread, NULL, warm_worker_run, &workers[i]) != 0) break;
        ++pool->started;
    }
    if (pool->started) {
        workers[0].state = WORKER_ASSIGNED;
        workers[0].count = workers[0].initial_count;
        pool->active = 1;
        atomic_store(&pool->idle, pool->started - 1);
        pthread_cond_broadcast(&pool->changed);
    }
    while (pool->started && !atomic_load(&pool->stop)) {
        if (chart_polish_now() >= w->polish_deadline) {
            pool->reason = NSAT_STOP_CANCELLED;
            atomic_store(&pool->stop, true);
        } else if (pool->waiting + atomic_load(&pool->idle) == pool->started && !pool->available) {
            if (pool->future && (pool->checkpoint == UINT64_MAX || pool->closed > pool->checkpoint)) {
                uint64_t grant = pool->future < pool->probe ? pool->future : pool->probe;
                pool->available += grant;
                pool->future -= grant;
                pool->checkpoint = pool->closed;
                ++pool->extensions;
            } else {
                pool->reason = NSAT_STOP_WORK;
                atomic_store(&pool->stop, true);
            }
            pthread_cond_broadcast(&pool->changed);
        }
        struct timespec wake;
        timespec_get(&wake, TIME_UTC);
        wake.tv_nsec += 1000000;
        if (wake.tv_nsec >= 1000000000) {
            ++wake.tv_sec;
            wake.tv_nsec -= 1000000000;
        }
        if (!atomic_load(&pool->stop)) pthread_cond_timedwait(&pool->changed, &pool->mutex, &wake);
    }
    pthread_cond_broadcast(&pool->changed);
    pthread_mutex_unlock(&pool->mutex);
    for (unsigned i = 0; i < pool->started; ++i) pthread_join(workers[i].thread, NULL);
    pthread_cond_destroy(&pool->changed);
    pthread_mutex_destroy(&pool->mutex);
    return pool->started;
}

#endif

IpcChartPolishResult ipc_chart_polish_decide(IpcChartPolish *state,
    const IpcChartPolishQuery *query, int *solution)
{
    IpcChartPolishResult result = {.status = NSAT_UNKNOWN};
    if (!state || !query || !solution || !query->allowed || !query->coverage_counts ||
        query->node_limit == 0 || query->max_row_degree < 1 ||
        query->forced_len < 0 || (query->forced_len && !query->forced) ||
        query->target < 2 || query->forced_len > query->target - 2) return result;
    WeightedCover *w = &state->weighted;
    const uint64_t before = w->budget.stats.work;
    if (w->budget.status != NSAT_OK || chart_polish_now() >= w->polish_deadline) return result;
    if (!state->ready && !chart_polish_root(state)) {
        result.work = w->budget.stats.work - before;
        result.peak_bytes = w->budget.stats.peak_memory_bytes;
        return result;
    }

    const size_t rows = state->input.positives, cols = w->result.maximal, words = w->mask_words;
    const size_t slots = (size_t)(query->target - query->forced_len);
    /* Each chosen column covers a new row, so the stack cannot exceed rows. */
    const size_t levels = slots < rows ? slots : rows;
    if (levels < 2) return result;
    const size_t minimum = warm_worker_fixed_bytes(w, query, levels) +
        rows * rows * sizeof(double) + sizeof(SavedBasis);
    const size_t available = nsat_memory_available(&w->budget);
    unsigned requested = state->workers;
    size_t affordable = minimum ? available / minimum : 0;
    if ((size_t)requested > affordable) requested = (unsigned)affordable;
    /* At most this many alternatives can be exposed at a pivot. Workers
       receive further descendants as they become available. */
    if (requested > (unsigned)query->max_row_degree) requested = (unsigned)query->max_row_degree;
    if ((uint64_t)requested > query->node_limit) requested = (unsigned)query->node_limit;
    if (!requested) goto finish;

    WarmWorker *workers = nsat_alloc(&w->budget, requested, sizeof(*workers));
    if (!workers) goto finish;
    memset(workers, 0, requested * sizeof(*workers));
    LiveBases live = {0};
    atomic_init(&live.used, 0);
    atomic_init(&live.peak, 0);
    unsigned prepared = 0;
    for (; prepared < requested;) {
        WarmWorker *worker = &workers[prepared++];
        worker->levels = levels;
        worker->query = query;
        if (!warm_worker_prepare(w, worker, &state->root, &live)) goto release;
    }
    live.capacity = nsat_memory_available(&w->budget);
    WarmWorker *first = &workers[0];
    all_rows(w, first->residuals);
    for (int i = 0; i < query->forced_len; ++i) {
        const uint64_t *mask = column(w, (size_t)query->forced[i]);
        for (size_t word = 0; word < words; ++word) first->residuals[word] &= ~mask[word];
    }
    if (next_row(first->residuals, words, 0) == SIZE_MAX) {
        memcpy(solution, query->forced, (size_t)query->forced_len * sizeof(int));
        result.status = NSAT_SAT;
        result.terms = query->forced_len;
        goto release;
    }
    for (size_t c = 0; c < cols; ++c)
        if (query->allowed[c]) first->choices[first->initial_count++] = c;

    WarmPool pool = {0};
    unsigned started = 0;
#if defined(HAVE_PTHREAD)
    if (requested > 1)
        started = warm_run_pool(&pool, workers, requested, w, query->node_limit);
#endif
    result.workers = started ? started : 1;
    const WarmWorker *winner = NULL;
    if (!started) {
        first->pool = NULL;
        first->weighted = *w;
        first->weighted.result.nodes = 0;
        first->weighted.budget.limits.context = &first->weighted;
        result.status = warm_polish_visit(first, 0, first->initial_count);
        w->budget = first->weighted.budget;
        w->budget.limits.context = w;
        w->progress = first->weighted.progress;
        w->checkpoint_progress = first->weighted.checkpoint_progress;
        w->next_probe = first->weighted.next_probe;
        result.nodes = first->weighted.result.nodes;
        if (result.status == NSAT_SAT) winner = first;
    } else {
        for (unsigned i = 0; i < started; ++i) {
            w->budget.stats.work += workers[i].weighted.budget.stats.work;
            result.nodes += workers[i].weighted.result.nodes;
        }
        w->progress += pool.closed;
        if (pool.extensions) w->checkpoint_progress = w->progress;
        w->next_probe = w->budget.limits.work - pool.future;
        if (pool.winner >= 0) {
            winner = &workers[pool.winner];
        } else if (pool.reason || pool.active) {
            nsat_stop(&w->budget, NSAT_UNKNOWN, pool.reason ? pool.reason : NSAT_STOP_CANCELLED);
        } else {
            result.status = NSAT_UNSAT;
        }
    }
    size_t peak = atomic_load(&live.peak) + w->budget.stats.memory_bytes;
    if (peak > w->budget.stats.peak_memory_bytes) w->budget.stats.peak_memory_bytes = peak;
    if (winner) {
        memcpy(solution, query->forced, (size_t)query->forced_len * sizeof(int));
        for (int i = 0; i < winner->length; ++i)
            solution[query->forced_len + i] = (int)winner->selected[i];
        result.terms = winner->length + query->forced_len;
        result.status = NSAT_SAT;
        ++w->progress;
    }

release:
    for (unsigned i = 0; i < prepared; ++i) warm_worker_release(w, &workers[i]);
    nsat_free(&w->budget, workers, requested * sizeof(*workers));
finish:
    result.work = w->budget.stats.work - before;
    result.peak_bytes = w->budget.stats.peak_memory_bytes;
    return result;
}
