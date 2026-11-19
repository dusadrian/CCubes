/* Included by main.c for the opt-in systematic projection build. */
#include <limits.h>
#include <mpi.h>
#include <pthread.h>

static int systematic_rank;
static int systematic_ranks = 1;
static bool systematic_workers_ready;

static void systematic_require(bool ok, const char *reason) {
    if (ok) return;
    fprintf(stderr, "Error: systematic MPI rank %d: %s\n", systematic_rank, reason);
    MPI_Abort(MPI_COMM_WORLD, 1);
    abort();
}

#include "systematic_sat_overlap.inc"

/* Use bounded MPI counts even for a large PI chart. Peer zero denotes
 * sending to the coordinator; a positive peer denotes receiving from that rank
 * on the coordinator side. Broadcast uses peer -1. */
static void systematic_transfer(void *data, size_t bytes, int peer) {
    unsigned char *next = data;
    while (bytes) {
        const int chunk = bytes > INT_MAX ? INT_MAX : (int)bytes;
        if (peer < 0) MPI_Bcast(next, chunk, MPI_BYTE, 0, MPI_COMM_WORLD);
        else if (systematic_rank) MPI_Send(next, chunk, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        else MPI_Recv(next, chunk, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        next += chunk;
        bytes -= (size_t)chunk;
    }
}

static void systematic_reserve(PIstorage *pi, int count, int words) {
    if (count <= pi->estimPI) return;
    const int grow = count - pi->estimPI;
    resize((void **)&pi->pichart_pos, TYPE_UINT64, grow, pi->estimPI, pi->pichart_words);
    resize((void **)&pi->implicants_pos, TYPE_UINT64, grow, pi->estimPI, words);
    resize((void **)&pi->implicants_val, TYPE_UINT64, grow, pi->estimPI, words);
    resize((void **)&pi->shared, TYPE_INT, grow, pi->estimPI, 1);
    resize((void **)&pi->covsum, TYPE_INT, grow, pi->estimPI, 1);
    resize((void **)&pi->covered, TYPE_INT, grow, pi->estimPI, 1);
    pi->estimPI = count;
}

static void systematic_records(PIstorage *pi, int first, int count, int words, int peer) {
    if (!count) return;
    systematic_transfer(pi->implicants_pos + (size_t)first * words,
                        (size_t)count * words * sizeof(uint64_t), peer);
    systematic_transfer(pi->implicants_val + (size_t)first * words,
                        (size_t)count * words * sizeof(uint64_t), peer);
    systematic_transfer(pi->pichart_pos + (size_t)first * pi->pichart_words,
                        (size_t)count * pi->pichart_words * sizeof(uint64_t), peer);
    systematic_transfer(pi->covsum + first, (size_t)count * sizeof(int), peer);
    systematic_transfer(pi->shared + first, (size_t)count * sizeof(int), peer);
}

static uint64_t systematic_boundary(uint64_t tasks, int rank) {
    /* Quotient/remainder avoids overflowing tasks * rank. */
    uint64_t quotient = tasks / (uint64_t)systematic_ranks;
    uint64_t remainder = tasks % (uint64_t)systematic_ranks;
    return quotient * (uint64_t)rank + ((uint64_t)rank < remainder ? (uint64_t)rank : remainder);
}

typedef struct {
    uint64_t first_task, end_task;
    int level, inputs, outputs, words;
    bool checkpointing, overlap;
} SystematicJob;

/* Earlier levels are immutable. Keep already collected current-level
 * candidates only on the coordinator, out of every rank's pruning index. */
static void systematic_prefix(PIWorkerContext *ctx, const int *level_start, int *base) {
    for (int o = 0; o < ctx->noutputs; ++o) {
        PIstorage *pi = &ctx->PInfo[o];
        int info[] = {pi->stop_search, 0, pi->ON_minterms, pi->OFF_minterms, pi->pichart_words};
        if (!systematic_rank && !pi->stop_search) info[1] = level_start[o];
        systematic_transfer(info, sizeof(info), -1);
        systematic_require(info[2] == pi->ON_minterms && info[3] == pi->OFF_minterms &&
                           info[4] == pi->pichart_words, "input shape differs across ranks");
        if (systematic_rank) {
            pi->stop_search = info[0];
            pi->foundPI = info[1];
            systematic_reserve(pi, pi->foundPI, ctx->implicant_words);
        }
        base[o] = pi->foundPI;
        systematic_records(pi, 0, info[1], ctx->implicant_words, -1);
    }
}

static void systematic_local_worker(uint64_t begin, uint64_t end, uint64_t stride,
                                    int id, int workers, void *data) {
    PIWorkerContext local = *(PIWorkerContext *)data;
    int max_shared = 0, multiplier = 0;
    local.max_shared = &max_shared;
    local.multiplier = &multiplier;
    local.checkpoint_deadline_sec = 0;
    pi_search_range_worker(begin, end, stride, id, workers, &local);
}

static void systematic_run_shard(PIWorkerContext *ctx, uint64_t first, uint64_t end, int threads) {
    uint64_t tasks = end - first;
    uint64_t begin = first + systematic_boundary(tasks, systematic_rank);
    uint64_t limit = first + systematic_boundary(tasks, systematic_rank + 1);
    systematic_require(begin == limit || ccubes_parallel_for(
        begin, limit, threads, true, systematic_local_worker, ctx
    ), "could not launch projection workers");
    systematic_require(!atomic_load(ctx->failed), "projection shard failed");
}

static double systematic_elapsed(const PIWorkerContext *ctx) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - ctx->start_time.tv_sec) +
           (double)(now.tv_nsec - ctx->start_time.tv_nsec) / 1e9;
}

/* Check the coordinator's clock between batches. One wave across all ranks
 * is the minimum useful granularity for large teams. Keep small teams from
 * paying for one collective per support. */
static uint64_t systematic_batches(PIWorkerContext *ctx, const SystematicJob *job, int threads) {
    uint64_t first = job->first_task;
    uint64_t batch_size = (uint64_t)systematic_ranks * (uint64_t)threads;
    if (batch_size < 32) batch_size = 32;

    for (;;) {
        uint64_t end = first;
        if (!systematic_rank) {
            bool save = false;
            if (job->checkpointing) {
                double elapsed = systematic_elapsed(ctx);
                bool hard_due = ctx->checkpoint_after_sec > 0.0 && elapsed >= ctx->checkpoint_after_sec;
                bool periodic_due = first > job->first_task && first < job->end_task &&
                                    elapsed >= ctx->checkpoint_deadline_sec;
                save = hard_due || periodic_due;
                if (save) atomic_store(ctx->time_up, true);
            }
            bool proved = false;
            if (job->overlap) {
                systematic_overlap_poll(ctx->PInfo, false);
                proved = systematic_overlap_all_active_proved(ctx->PInfo);
            }
            if (!save && !proved && first < job->end_task) {
                uint64_t remaining = job->end_task - first;
                end = (job->checkpointing || job->overlap) && remaining > batch_size
                    ? first + batch_size : job->end_task;
            }
        }
        systematic_transfer(&end, sizeof(end), -1);
        if (end == first) return first;
        systematic_run_shard(ctx, first, end, threads);
        if (job->checkpointing || job->overlap) MPI_Barrier(MPI_COMM_WORLD);
        first = end;
    }
}

static void systematic_follow(PIWorkerContext *ctx, int threads) {
    for (;;) {
        SystematicJob job;
        systematic_transfer(&job, sizeof(job), -1);
        if (!job.level) return;
        systematic_require(job.inputs == ctx->ninputs && job.outputs == ctx->noutputs &&
                           job.words == ctx->implicant_words, "input shape differs across ranks");
        ctx->k = job.level;
        int base[ctx->noutputs];
        systematic_prefix(ctx, NULL, base);
        systematic_require(build_pi_coverage_indices(&ctx->coverage_indices, ctx->PInfo,
            ctx->noutputs, base, ctx->implicant_words), "could not build the completed-level index");
        systematic_batches(ctx, &job, threads);
        uint64_t rejected = 0;
        for (int o = 0; o < ctx->noutputs; ++o)
            rejected += atomic_load(&ctx->coverage_indices[o].subsumption_rejections);
        destroy_pi_coverage_indices(ctx->coverage_indices, ctx->noutputs);
        for (int o = 0; o < ctx->noutputs; ++o) {
            PIstorage *pi = &ctx->PInfo[o];
            int count = pi->foundPI - base[o];
            systematic_transfer(&count, sizeof(count), 0);
            systematic_records(pi, base[o], count, ctx->implicant_words, 0);
        }
        systematic_transfer(&rejected, sizeof(rejected), 0);
    }
}

static uint64_t systematic_dispatch(PIWorkerContext *ctx, const int *level_start,
                                    uint64_t first, uint64_t end, int threads) {
    SystematicJob job = {
        .first_task = first, .end_task = end, .level = ctx->k, .inputs = ctx->ninputs,
        .outputs = ctx->noutputs, .words = ctx->implicant_words,
        .checkpointing = ctx->checkpoint_deadline_sec > 0.0,
        .overlap = systematic_overlap.running >= 0 && systematic_overlap.jobs != NULL
    };
    const double started = MPI_Wtime();
    systematic_transfer(&job, sizeof(job), -1);
    int base[ctx->noutputs];
    systematic_prefix(ctx, level_start, base);
    uint64_t frontier = systematic_batches(ctx, &job, threads);
    for (int o = 0; o < ctx->noutputs; ++o) {
        const PIstorage *pi = &ctx->PInfo[o];
        for (int i = base[o]; i < pi->foundPI; ++i)
            if (pi->shared[i] > *ctx->max_shared) *ctx->max_shared = pi->shared[i];
    }
    uint64_t remote_records = 0;
    uint64_t subsumed = 0;
    for (int rank = 1; rank < systematic_ranks; ++rank) {
        for (int o = 0; o < ctx->noutputs; ++o) {
            PIstorage *pi = &ctx->PInfo[o];
            int count;
            systematic_transfer(&count, sizeof(count), rank);
            systematic_require(count >= 0 && count <= INT_MAX - pi->foundPI,
                               "merged PI count exceeds the chart format");
            if (count) {
                int first = pi->foundPI;
                systematic_reserve(pi, first + count, ctx->implicant_words);
                systematic_records(pi, first, count, ctx->implicant_words, rank);
                pi->foundPI += count;
                for (int i = first; i < pi->foundPI; ++i)
                    if (pi->shared[i] > *ctx->max_shared) *ctx->max_shared = pi->shared[i];
            }
            remote_records += (uint64_t)count;
        }
        uint64_t rejected;
        systematic_transfer(&rejected, sizeof(rejected), rank);
        subsumed += rejected;
    }
    if (job.overlap && frontier < end && !atomic_load(ctx->time_up)) {
        /* Every active output is proved. Discard the unfinished level as a
         * unit: changing its participating outputs would change sharing. */
        for (int o = 0; o < ctx->noutputs; ++o)
            ctx->PInfo[o].foundPI = base[o];
        systematic_overlap_apply_stops(ctx->PInfo, ctx->k);
    }
    fprintf(stderr, "CCUBES_SYSTEMATIC_MPI level=%d ranks=%d first=%llu frontier=%llu tasks=%llu "
            "remote_records=%llu remote_subsumed=%llu seconds=%.6f\n",
            ctx->k, systematic_ranks, (unsigned long long)first, (unsigned long long)frontier,
            (unsigned long long)end, (unsigned long long)remote_records,
            (unsigned long long)subsumed, MPI_Wtime() - started);
    return frontier;
}


static uint64_t systematic_hash(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; ++i)
        hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static void systematic_trace_chart(const PIstorage *pi, int level, int output, int first, int words) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = first; i < pi->foundPI; ++i) {
        hash = systematic_hash(hash, pi->implicants_pos + (size_t)i * words, (size_t)words * sizeof(uint64_t));
        hash = systematic_hash(hash, pi->implicants_val + (size_t)i * words, (size_t)words * sizeof(uint64_t));
        hash = systematic_hash(hash, pi->pichart_pos + (size_t)i * pi->pichart_words,
                               (size_t)pi->pichart_words * sizeof(uint64_t));
        hash = systematic_hash(hash, pi->shared + i, sizeof(int));
        hash = systematic_hash(hash, pi->covsum + i, sizeof(int));
    }
    fprintf(stderr, "CCUBES_SYSTEMATIC_MPI_CHART level=%d output=%d count=%d hash=%016llx\n",
            level, output + 1, pi->foundPI - first, (unsigned long long)hash);
}
