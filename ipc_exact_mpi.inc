#include <mpi.h>
#include <stdio.h>

extern void ccubes_mpi_exact_start(void);

typedef struct {
    int inputs, on_rows, off_rows, level, capacity, excluded;
    WeightedMMCSPriceOptions options;
} ExactMPIJob;

typedef struct {
    WeightedMMCSPriceResult price;
    WeightedMMCSPriceStatus status;
    int count, begin, end;
} ExactMPIResult;

static int exact_mpi_bytes(size_t n, size_t width) {
    if (n > (size_t)INT_MAX / width) MPI_Abort(MPI_COMM_WORLD, 1);
    return (int)(n * width);
}

static void *exact_mpi_allocate(size_t n, size_t width) {
    exact_mpi_bytes(n, width);
    void *data = calloc(n ? n : 1, width);
    if (!data) MPI_Abort(MPI_COMM_WORLD, 1);
    return data;
}

static void exact_mpi_bcast(void *data, size_t n, size_t width) {
    MPI_Bcast(data, exact_mpi_bytes(n, width), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void exact_mpi_chart(const PIstorage *pi, const double *weights) {
    exact_mpi_bcast(pi->ON_set, (size_t)pi->inputs * pi->ON_minterms, sizeof(int));
    exact_mpi_bcast(pi->OFF_set, (size_t)pi->inputs * pi->OFF_minterms, sizeof(int));
    exact_mpi_bcast((void *)weights, pi->ON_minterms, sizeof(double));
}

static WeightedMMCSPriceBatch exact_mpi_batch(int capacity, int inputs, int words) {
    return (WeightedMMCSPriceBatch){
        .capacity = capacity,
        .weights = exact_mpi_allocate(capacity, sizeof(double)),
        .anchors = exact_mpi_allocate(capacity, sizeof(int)),
        .support_sizes = exact_mpi_allocate(capacity, sizeof(int)),
        .vertices = exact_mpi_allocate((size_t)capacity * inputs, sizeof(int)),
        .coverage = exact_mpi_allocate((size_t)capacity * words, sizeof(uint64_t))
    };
}

static void exact_mpi_batch_free(WeightedMMCSPriceBatch *batch) {
    free(batch->weights);
    free(batch->anchors);
    free(batch->support_sizes);
    free(batch->vertices);
    free(batch->coverage);
}

static void exact_mpi_transfer(void *data, size_t n, size_t width, int rank) {
    const int bytes = exact_mpi_bytes(n, width);
    if (rank) MPI_Recv(data, bytes, MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else MPI_Send(data, bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
}

static void exact_mpi_batch_transfer(
    WeightedMMCSPriceBatch *batch, int inputs, int words, int rank
) {
    int n = batch->count;
    exact_mpi_transfer(batch->weights, n, sizeof(double), rank);
    exact_mpi_transfer(batch->anchors, n, sizeof(int), rank);
    exact_mpi_transfer(batch->support_sizes, n, sizeof(int), rank);
    exact_mpi_transfer(batch->vertices, (size_t)n * inputs, sizeof(int), rank);
    exact_mpi_transfer(batch->coverage, (size_t)n * words, sizeof(uint64_t), rank);
}


static ExactMPIResult exact_mpi_shard(
    const PIstorage *pi, const WeightedMMCSPreparedRows *prepared,
    const double *weights, int level, WeightedMMCSPriceOptions options,
    int *vertices, WeightedMMCSPriceBatch *batch, int rank, int ranks
) {
    const int first = options.shard_begin;
    const int count = (options.shard_end ? options.shard_end : pi->ON_minterms) - first;
    options.shard_begin = first + (int)((int64_t)count * rank / ranks);
    options.shard_end = first + (int)((int64_t)count * (rank + 1) / ranks);
    ExactMPIResult result = {.begin = options.shard_begin, .end = options.shard_end};
    if (options.shard_begin == options.shard_end) {
        result.price.best_anchor = -1;
        result.price.best_weight = -INFINITY;
        result.price.anchor_sweep_complete = true;
        return result;
    }
    result.status = bounded_mmcs_price_output_batch_prepared(
        pi, prepared, level, weights, vertices, batch, &options, &result.price);
    result.count = batch ? batch->count : 0;
    return result;
}

void ccubes_mpi_exact_follow(void) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    ExactMPIJob job;
    exact_mpi_bcast(&job, 1, sizeof(job));
    PIstorage pi = {.inputs = job.inputs, .outputs = 1, .cov_bits = 64,
        .ON_minterms = job.on_rows, .OFF_minterms = job.off_rows,
        .pichart_words = (job.on_rows + 63) / 64};
    pi.ON_set = exact_mpi_allocate((size_t)job.inputs * job.on_rows, sizeof(int));
    pi.OFF_set = exact_mpi_allocate((size_t)job.inputs * job.off_rows, sizeof(int));
    double *weights = exact_mpi_allocate(job.on_rows, sizeof(double));
    int *vertices = exact_mpi_allocate(job.inputs, sizeof(int));
    exact_mpi_chart(&pi, weights);
    /* Anchor the local clock before the coordinator measures remaining
     * time. The barrier makes translation conservative even with transport
     * delay; receipt of the duration must not restart the time slice. */
    const uint64_t deadline_origin = monotonic_nanoseconds();
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t remaining;
    exact_mpi_bcast(&remaining, 1, sizeof(remaining));
    job.options.deadline_nanoseconds = remaining == UINT64_MAX ? 0
        : deadline_origin + remaining;
    WeightedMMCSPreparedRows prepared;
    if (!bounded_mmcs_prepare_binary_rows(&pi, job.inputs, &prepared))
        MPI_Abort(MPI_COMM_WORLD, 1);
    CoverageExclusion exclusion = {0};
    uint64_t *excluded = exact_mpi_allocate((size_t)job.excluded * pi.pichart_words, sizeof(uint64_t));
    exact_mpi_bcast(excluded, (size_t)job.excluded * pi.pichart_words, sizeof(uint64_t));
    if (!coverage_exclusion_init(&exclusion, excluded, job.excluded, pi.pichart_words))
        MPI_Abort(MPI_COMM_WORLD, 1);
    WeightedMMCSPriceBatch batch = exact_mpi_batch(job.capacity, job.inputs, pi.pichart_words);
    batch.exclusion = exclusion.slots ? &exclusion : NULL;
    ExactMPIResult result = exact_mpi_shard(&pi, &prepared, weights, job.level, job.options,
        vertices, job.capacity ? &batch : NULL, rank, ranks);
    exact_mpi_transfer(&result, 1, sizeof(result), 0);
    exact_mpi_transfer(vertices, job.inputs, sizeof(int), 0);
    if (job.capacity) exact_mpi_batch_transfer(&batch, job.inputs, pi.pichart_words, 0);
    exact_mpi_batch_free(&batch);
    free(exclusion.slots);
    free(excluded);
    bounded_mmcs_prepared_rows_destroy(&prepared);
    free(pi.ON_set);
    free(pi.OFF_set);
    free(weights);
    free(vertices);
}

WeightedMMCSPriceStatus ccubes_mpi_exact_price(
    const PIstorage *pi, const WeightedMMCSPreparedRows *prepared, int level,
    const double *weights, int *vertices, WeightedMMCSPriceBatch *batch,
    const WeightedMMCSPriceOptions *options, WeightedMMCSPriceResult *result
) {
    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const double started = MPI_Wtime();
    WeightedMMCSPriceStatus status;
    if (ranks == 1 || options->anchor_limit) {
        status = bounded_mmcs_price_output_batch_prepared(
            pi, prepared, level, weights, vertices, batch, options, result);
        goto finished;
    }
    ccubes_mpi_exact_start();
    ExactMPIJob job = {pi->inputs, pi->ON_minterms, pi->OFF_minterms,
        level, batch ? batch->capacity : 0,
        batch && batch->exclusion ? batch->exclusion->count : 0, *options};
    job.options.deadline_nanoseconds = 0; /* Monotonic stamps are host-local. */
    exact_mpi_bcast(&job, 1, sizeof(job));
    exact_mpi_chart(pi, weights);
    MPI_Barrier(MPI_COMM_WORLD);
    const uint64_t now = monotonic_nanoseconds();
    uint64_t remaining = !options->deadline_nanoseconds ? UINT64_MAX
        : options->deadline_nanoseconds > now ? options->deadline_nanoseconds - now : 0;
    exact_mpi_bcast(&remaining, 1, sizeof(remaining));
    exact_mpi_bcast(job.excluded ? (void *)batch->exclusion->coverage : NULL,
        (size_t)job.excluded * pi->pichart_words, sizeof(uint64_t));
    WeightedPriceWorker *sources = exact_mpi_allocate(ranks, sizeof(*sources));
    int *remote = exact_mpi_allocate(pi->inputs, sizeof(int));
    sources[0].local_batch = exact_mpi_batch(job.capacity, pi->inputs, pi->pichart_words);
    sources[0].local_batch.exclusion = batch ? batch->exclusion : NULL;
    ExactMPIResult local = exact_mpi_shard(pi, prepared, weights, level, *options,
        vertices, job.capacity ? &sources[0].local_batch : NULL, 0, ranks);
    *result = local.price;
    status = local.status;
    for (int rank = 1; rank < ranks; ++rank) {
        ExactMPIResult reply;
        exact_mpi_transfer(&reply, 1, sizeof(reply), rank);
        exact_mpi_transfer(remote, pi->inputs, sizeof(int), rank);
        if (job.capacity) {
            WeightedMMCSPriceBatch *incoming = &sources[rank].local_batch;
            if (reply.count < 0 || reply.count > job.capacity) MPI_Abort(MPI_COMM_WORLD, 1);
            *incoming = exact_mpi_batch(reply.count, pi->inputs, pi->pichart_words);
            incoming->count = reply.count;
            exact_mpi_batch_transfer(incoming, pi->inputs, pi->pichart_words, rank);
        }
        const WeightedMMCSPriceResult *p = &reply.price;
        if (reply.status > status) status = reply.status;
        result->anchors_completed += p->anchors_completed;
        result->anchors_total += p->anchors_total;
        result->search_nodes += p->search_nodes;
        result->completed_transversals += p->completed_transversals;
        result->weight_prunes += p->weight_prunes;
        add_price_telemetry(&result->telemetry, &p->telemetry);
        if (p->estimated_bytes_per_worker > result->estimated_bytes_per_worker)
            result->estimated_bytes_per_worker = p->estimated_bytes_per_worker;
        if (p->peak_incidence_entries > result->peak_incidence_entries)
            result->peak_incidence_entries = p->peak_incidence_entries;
        result->effective_threads += p->effective_threads;
        result->max_weight_upper_bound = fmax(result->max_weight_upper_bound, p->max_weight_upper_bound);
        result->anchor_sweep_complete &= p->anchor_sweep_complete;
        if (p->best_anchor >= 0 && (result->best_anchor < 0 || p->best_weight > result->best_weight ||
            (p->best_weight == result->best_weight && weighted_geometry_precedes(
                p->best_anchor, p->best_support_size, remote, result->best_anchor,
                result->best_support_size, vertices, pi->inputs)))) {
            result->best_weight = p->best_weight;
            result->best_anchor = p->best_anchor;
            result->best_support_size = p->best_support_size;
            memcpy(vertices, remote, (size_t)p->best_support_size * sizeof(int));
        }
    }
    if (batch && status <= WEIGHTED_MMCS_PRICE_LIMIT_REACHED &&
        !weighted_merge_local_batches(sources, ranks, batch, pi->inputs, pi->pichart_words))
        status = WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
    result->limit_reached = status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED;
    for (int rank = 0; rank < ranks; ++rank) exact_mpi_batch_free(&sources[rank].local_batch);
    free(sources);
    free(remote);
finished:
    fprintf(stderr, "CCUBES_MPI_EXACT ranks=%d anchors=%d/%d nodes=%llu complete=%d "
        "threads=%d upper=%.17g best=%.17g seconds=%.6f\n",
        ranks, result->anchors_completed, result->anchors_total,
        (unsigned long long)result->search_nodes, result->anchor_sweep_complete,
        result->effective_threads, result->max_weight_upper_bound, result->best_weight,
        MPI_Wtime() - started);
    return status;
}
