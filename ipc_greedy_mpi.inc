/* Included only in the experimental MPI build. Rank zero prepares each batch
 * and owns the master. Ranks construct disjoint slots of the batch. */
#include <mpi.h>
#include <stdio.h>

typedef struct {
    int candidates, anchors, variants, beam, inputs, on, off, on_words;
    int off_words, support, threads, cleanup;
    double weight, threshold, tolerance;
} GreedyMPIWork;

static void greedy_mpi_require(bool ok) {
    if (!ok) {
        fprintf(stderr, "Distributed greedy pricing failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        abort();
    }
}

static int greedy_mpi_bytes(size_t count, size_t size) {
    greedy_mpi_require(count <= (size_t)INT_MAX / size);
    return (int)(count * size);
}

static void greedy_mpi_bcast(const void *data, size_t count, size_t size) {
    if (count)
        MPI_Bcast((void *)data, greedy_mpi_bytes(count, size), MPI_BYTE, 0, MPI_COMM_WORLD);
}

/* Ranks exchange the same contiguous arrays in the same sequence. The
 * coordinator's original compaction/fold retains global slot order. */
static void greedy_mpi_results(GreedyWorkerContext *context, int split, int end,
                               int rank, bool coordinator) {
    size_t count = (size_t)(end - split);
#define TRANSFER(array, width) do { \
    size_t stride = (size_t)(width); \
    int bytes = greedy_mpi_bytes(count * stride, sizeof(*(array))); \
    void *data = (array) + (size_t)split * stride; \
    if (coordinator) MPI_Recv(data, bytes, MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); \
    else MPI_Send(data, bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD); \
} while (0)
    TRANSFER(context->attempted, 1);
    TRANSFER(context->valid, 1);
    TRANSFER(context->literal_evaluations, 1);
    TRANSFER(context->batch->weights, 1);
    TRANSFER(context->batch->anchors, 1);
    TRANSFER(context->batch->support_sizes, 1);
    TRANSFER(context->batch->vertices, context->ninputs);
    TRANSFER(context->batch->coverage, context->on_words);
#undef TRANSFER
}

static int greedy_parallel_construct(
    uint64_t start, uint64_t end, int threads, bool strided,
    ccubes_range_worker worker, void *data
) {
    GreedyWorkerContext *context = data;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /* An absolute monotonic deadline is local to each host. Leave existing
     * timed calls on the coordinator. Native discovery batches are atomic. */
    if (size == 1 || end < 2 || context->deadline_nanoseconds)
        return ccubes_parallel_for(start, end, threads, strided, worker, data);
    greedy_mpi_require(start == 0 && end == (uint64_t)context->candidate_count);
    GreedyMPIWork job = {
        context->candidate_count, context->anchor_count, context->variants_per_anchor,
        context->prefix_beam_width, context->ninputs, context->on_rows, context->off_rows,
        context->on_words, context->off_words, context->max_support, threads,
        context->cleanup_prefix != NULL, context->initial_weight, context->threshold,
        context->tolerance
    };
    const double started = MPI_Wtime();
    greedy_mpi_bcast(&job, 1, sizeof(job));
    greedy_mpi_bcast(context->pi->ON_set, (size_t)job.on * job.inputs, sizeof(int));
    greedy_mpi_bcast(context->prepared->on_columns, (size_t)job.on_words * job.inputs, sizeof(uint64_t));
    greedy_mpi_bcast(context->prepared->off_columns, (size_t)job.off_words * job.inputs, sizeof(uint64_t));
    greedy_mpi_bcast(context->row_weights, job.on, sizeof(double));
    greedy_mpi_bcast(context->root_scores, (size_t)job.inputs * 2, sizeof(GreedyRootScore));
    greedy_mpi_bcast(context->anchors, job.anchors, sizeof(GreedyAnchor));
    greedy_mpi_bcast(context->prefix_beam_vertices, (size_t)job.anchors * job.beam, sizeof(int));
    const int split = job.candidates / size;
    const double dispatched = MPI_Wtime();
    if (split)
        greedy_mpi_require(ccubes_parallel_for(0, (uint64_t)split, threads, strided, worker, data));
    const double local_done = MPI_Wtime();
    for (int rank = 1; rank < size; ++rank)
        greedy_mpi_results(context, (int)((int64_t)job.candidates * rank / size),
                           (int)((int64_t)job.candidates * (rank + 1) / size), rank, true);
    fprintf(stderr, "CCUBES_MPI_DISCOVERY candidates=%d local=%d remote=%d "
            "dispatch_seconds=%.6f local_seconds=%.6f gather_seconds=%.6f seconds=%.6f\n",
            job.candidates, split, job.candidates - split, dispatched - started,
            local_done - dispatched, MPI_Wtime() - local_done, MPI_Wtime() - started);
    return 1;
}

static void *greedy_mpi_alloc(size_t count, size_t size) {
    greedy_mpi_bytes(count, size);
    void *data = calloc(count ? count : 1, size);
    greedy_mpi_require(data != NULL);
    return data;
}

extern void ccubes_mpi_exact_follow(void);

void ccubes_mpi_exact_start(void) {
    GreedyMPIWork marker = {.candidates = -1};
    greedy_mpi_bcast(&marker, 1, sizeof(marker));
}

static void greedy_mpi_follow(int rank, int size) {
    ProgressiveGreedyPriceWorkspace workspace = {0};
    for (;;) {
        GreedyMPIWork job;
        greedy_mpi_bcast(&job, 1, sizeof(job));
        if (!job.candidates) break;
        if (job.candidates == -1) {
            ccubes_mpi_exact_follow();
            continue;
        }
        PIstorage pi = {.inputs = job.inputs, .ON_minterms = job.on, .OFF_minterms = job.off};
        WeightedMMCSPreparedRows prepared = {0};
        const int threads = ccubes_effective_thread_count(job.threads);
        greedy_mpi_require(greedy_workspace_prepare(
            &workspace, job.inputs, job.on, job.on_words, job.off_words,
            threads, job.candidates, (size_t)job.anchors * job.beam, job.cleanup));
        pi.ON_set = greedy_mpi_alloc((size_t)job.on * job.inputs, sizeof(int));
        prepared.on_columns = greedy_mpi_alloc((size_t)job.on_words * job.inputs, sizeof(uint64_t));
        prepared.off_columns = greedy_mpi_alloc((size_t)job.off_words * job.inputs, sizeof(uint64_t));
        double *weights = greedy_mpi_alloc(job.on, sizeof(double));
        greedy_mpi_bcast(pi.ON_set, (size_t)job.on * job.inputs, sizeof(int));
        greedy_mpi_bcast(prepared.on_columns, (size_t)job.on_words * job.inputs, sizeof(uint64_t));
        greedy_mpi_bcast(prepared.off_columns, (size_t)job.off_words * job.inputs, sizeof(uint64_t));
        greedy_mpi_bcast(weights, job.on, sizeof(double));
        greedy_mpi_bcast(workspace.root_scores, (size_t)job.inputs * 2, sizeof(GreedyRootScore));
        greedy_mpi_bcast(workspace.anchors, job.anchors, sizeof(GreedyAnchor));
        greedy_mpi_bcast(workspace.prefix_beam_vertices, (size_t)job.anchors * job.beam, sizeof(int));
        WeightedMMCSPriceBatch batch = {
            .capacity = job.candidates,
            .weights = greedy_mpi_alloc(job.candidates, sizeof(double)),
            .anchors = greedy_mpi_alloc(job.candidates, sizeof(int)),
            .support_sizes = greedy_mpi_alloc(job.candidates, sizeof(int)),
            .vertices = greedy_mpi_alloc((size_t)job.candidates * job.inputs, sizeof(int)),
            .coverage = greedy_mpi_alloc((size_t)job.candidates * job.on_words, sizeof(uint64_t))
        };
        GreedyWorkerContext context = {
            .pi = &pi, .prepared = &prepared, .row_weights = weights,
            .root_scores = workspace.root_scores, .anchors = workspace.anchors,
            .anchor_count = job.anchors, .variants_per_anchor = job.variants,
            .prefix_beam_width = job.beam, .prefix_beam_vertices = workspace.prefix_beam_vertices,
            .candidate_count = job.candidates, .ninputs = job.inputs, .on_rows = job.on,
            .off_rows = job.off, .on_words = job.on_words, .off_words = job.off_words,
            .max_support = job.support, .initial_weight = job.weight,
            .threshold = job.threshold, .tolerance = job.tolerance,
            .batch = &batch, .attempted = workspace.attempted, .valid = workspace.valid,
            .literal_evaluations = workspace.literal_evaluations, .remaining = workspace.remaining,
            .compatible = workspace.compatible, .blocked = workspace.blocked,
            .cleanup_prefix = workspace.cleanup_prefix, .selected = workspace.selected,
            .selection_order = workspace.selection_order
        };
        const int begin = (int)((int64_t)job.candidates * rank / size);
        const int end = (int)((int64_t)job.candidates * (rank + 1) / size);
        if (begin < end)
            greedy_mpi_require(ccubes_parallel_for((uint64_t)begin, (uint64_t)end,
                threads, false, greedy_construct_range, &context));
        greedy_mpi_results(&context, begin, end, rank, false);
        free(pi.ON_set);
        free(prepared.on_columns);
        free(prepared.off_columns);
        free(weights);
        free(batch.weights);
        free(batch.anchors);
        free(batch.support_sizes);
        free(batch.vertices);
        free(batch.coverage);
    }
    greedy_workspace_release(&workspace);
}

static uint64_t greedy_mpi_hash(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; ++i)
        hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static void greedy_mpi_fingerprint(
    const WeightedMMCSPriceBatch *batch, int inputs, int words,
    const ProgressiveGreedyPriceStats *stats
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const size_t count = (size_t)batch->count;
    hash = greedy_mpi_hash(hash, batch->weights, count * sizeof(double));
    hash = greedy_mpi_hash(hash, batch->anchors, count * sizeof(int));
    hash = greedy_mpi_hash(hash, batch->support_sizes, count * sizeof(int));
    for (size_t slot = 0; slot < count; ++slot)
        hash = greedy_mpi_hash(hash, batch->vertices + slot * inputs,
                               (size_t)batch->support_sizes[slot] * sizeof(int));
    hash = greedy_mpi_hash(hash, batch->coverage, count * words * sizeof(uint64_t));
    fprintf(stderr, "CCUBES_MPI_BATCH count=%d hash=%016llx anchors=%llu "
            "literals=%llu scout_literals=%llu\n", batch->count, (unsigned long long)hash,
            (unsigned long long)stats->anchors_attempted,
            (unsigned long long)stats->literal_evaluations,
            (unsigned long long)stats->scout_literal_evaluations);
}

/* main.c is compiled with its entry point renamed in this isolated build. */
extern int ccubes_mpi_main(int argc, char **argv);
int main(int argc, char **argv) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    greedy_mpi_require(size >= 1 && provided >= MPI_THREAD_FUNNELED);
    int status = 0;
    if (rank == 0) {
        status = ccubes_mpi_main(argc, argv);
        if (size > 1) {
            GreedyMPIWork stop = {0};
            greedy_mpi_bcast(&stop, 1, sizeof(stop));
        }
    } else {
        greedy_mpi_follow(rank, size);
    }
    MPI_Finalize();
    return status;
}
