/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdatomic.h>
#include <errno.h>
#include <math.h>
#include "main.h"
#include "cli.h"
#include "runtime_options.h"
#include "certified_stop.h"
#include "checkpoint.h"
#include "file_replace.h"
#include "lock_stats.h"
#include "ccubes_threads.h"
#include "pool_selection.h"
#include "plateau_probe.h"
#include "bounded_mmcs.h"
#include "progressive_cover.h"
#include "progressive_column_pool.h"
#include "progressive_input.h"
#include "progressive_rwls.h"

#define CCUBES_AUTO_MMCS_MIN_PROJECTION_TASKS UINT64_C(8192)
#define CCUBES_AUTO_MMCS_MIN_NODE_BUDGET UINT64_C(8192)
/*
 * Ceiling on the break-even budget.  Break-even alone would let a hopeless
 * trial run as long as projection would; this caps the wasted effort while
 * still being ~10^4 times the old fixed budget.
 */
#define CCUBES_AUTO_MMCS_MAX_NODE_LIMIT UINT64_C(100000000)
#define CCUBES_AUTO_MMCS_MAX_TRIAL_THREADS 4

#define CCUBES_INITIAL_PI_MEMORY_BYTES UINT64_C(268435456)
#define CCUBES_INITIAL_PI_CAPACITY_MIN 64
#define CCUBES_INITIAL_PI_CAPACITY_MAX 8192
#define CCUBES_PI_CAPACITY_GROWTH 1024

static int ordinary_initial_pi_capacity(
    const PIstorage *outputs,
    int noutputs,
    int implicant_words,
    uint64_t *estimated_bytes,
    int *active_outputs,
    int *constant_zero_outputs,
    int *constant_one_outputs
) {
    uint64_t bytes_per_slot = 0;
    int active = 0;
    int constant_zero = 0;
    int constant_one = 0;
    for (int output = 0; output < noutputs; ++output) {
        const PIstorage *pi = &outputs[output];
        if (pi->ON_minterms <= 0) {
            constant_zero++;
            continue;
        }
        if (pi->OFF_minterms <= 0) {
            constant_one++;
            continue;
        }
        active++;
        uint64_t column_bytes =
            (uint64_t)pi->pichart_words * sizeof(uint64_t) +
            (uint64_t)implicant_words * 2u * sizeof(uint64_t) +
            3u * sizeof(int);
        if (UINT64_MAX - bytes_per_slot < column_bytes) {
            bytes_per_slot = UINT64_MAX;
            break;
        }
        bytes_per_slot += column_bytes;
    }

    uint64_t capacity = bytes_per_slot > 0
        ? CCUBES_INITIAL_PI_MEMORY_BYTES / bytes_per_slot
        : CCUBES_INITIAL_PI_CAPACITY_MIN;
    if (capacity < CCUBES_INITIAL_PI_CAPACITY_MIN) {
        capacity = CCUBES_INITIAL_PI_CAPACITY_MIN;
    }
    if (capacity > CCUBES_INITIAL_PI_CAPACITY_MAX) {
        capacity = CCUBES_INITIAL_PI_CAPACITY_MAX;
    }
    if (estimated_bytes) {
        *estimated_bytes = bytes_per_slot == UINT64_MAX
            ? UINT64_MAX
            : bytes_per_slot * capacity;
    }
    if (active_outputs) *active_outputs = active;
    if (constant_zero_outputs) *constant_zero_outputs = constant_zero;
    if (constant_one_outputs) *constant_one_outputs = constant_one;
    return (int)capacity;
}

static long long written_pla_product_count(const char *path) {
    if (!path) return -1;
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    char line[256];
    long long products = -1;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, ".p %lld", &products) == 1) break;
    }
    fclose(file);
    return products;
}

typedef struct {
    const char *output_path;
} ProgressiveIncumbentFileContext;

static bool persist_progressive_incumbent(
    int ninputs,
    int cardinality,
    const int *solution,
    int selected_horizon,
    int generation,
    void *data
) {
    ProgressiveIncumbentFileContext *context =
        (ProgressiveIncumbentFileContext *)data;
    if (
        !context ||
        !context->output_path ||
        ninputs <= 0 ||
        cardinality <= 0 ||
        !solution ||
        generation <= 0
    ) {
        return false;
    }

    size_t path_length = strlen(context->output_path);
    char *temporary_path = (char *)malloc(path_length + 16u);
    char *metadata_path = (char *)malloc(path_length + 16u);
    char *temporary_metadata_path = (char *)malloc(path_length + 32u);
    if (!temporary_path || !metadata_path || !temporary_metadata_path) {
        free(temporary_path);
        free(metadata_path);
        free(temporary_metadata_path);
        return false;
    }
    snprintf(
        temporary_path,
        path_length + 16u,
        "%s.inc.tmp",
        context->output_path
    );
    snprintf(
        metadata_path,
        path_length + 16u,
        "%s.meta",
        context->output_path
    );
    snprintf(
        temporary_metadata_path,
        path_length + 32u,
        "%s.inc.tmp",
        metadata_path
    );

    if (!run_log_check_path(context->output_path) ||
        !run_log_check_path(temporary_path) ||
        !run_log_check_path(metadata_path) ||
        !run_log_check_path(temporary_metadata_path)) {
        free(temporary_path);
        free(metadata_path);
        free(temporary_metadata_path);
        return false;
    }

    bool ok = false;
    FILE *pla = fopen(temporary_path, "w");
    if (!pla) goto DONE;
    if (
        fprintf(
            pla,
            ".i %d\n.o 1\n.p %d\n",
            ninputs,
            cardinality
        ) < 0
    ) {
        fclose(pla);
        goto DONE;
    }
    for (int cube = 0; cube < cardinality; ++cube) {
        for (int input = 0; input < ninputs; ++input) {
            int value = solution[
                (size_t)input * (size_t)cardinality + (size_t)cube
            ];
            char symbol = value == 0 ? '-' : (value == 1 ? '0' : '1');
            if (fputc(symbol, pla) == EOF) {
                fclose(pla);
                goto DONE;
            }
        }
        if (fputs(" 1\n", pla) == EOF) {
            fclose(pla);
            goto DONE;
        }
    }
    bool pla_write_failed = fputs(".e\n", pla) == EOF;
    if (fclose(pla) != 0) pla_write_failed = true;
    if (pla_write_failed) goto DONE;

    FILE *metadata = fopen(temporary_metadata_path, "w");
    if (!metadata) goto DONE;
    bool metadata_write_failed = fprintf(
        metadata,
        "status=validated-incumbent\n"
        "generation=%d\n"
        "cardinality=%d\n"
        "selected_horizon=%d\n",
        generation,
        cardinality,
        selected_horizon
    ) < 0;
    if (fclose(metadata) != 0) metadata_write_failed = true;
    if (metadata_write_failed) goto DONE;
    if (
        !ccubes_replace_file(temporary_path, context->output_path) ||
        !ccubes_replace_file(temporary_metadata_path, metadata_path)
    ) {
        goto DONE;
    }
    ok = true;

DONE:
    if (!ok) {
        remove(temporary_path);
        remove(temporary_metadata_path);
    }
    free(temporary_path);
    free(metadata_path);
    free(temporary_metadata_path);
    return ok;
}

/*
 * Keep the ordinary sharing-aware incumbent in a boundary pool.  Pool
 * enumeration deliberately skips column dominance so that locally
 * interchangeable cubes remain available for cross-output coordination; that
 * makes it slower and can leave a weaker incumbent under bounded effort.
 * Seeding the pool with the ordinary solution makes pooling a secondary
 * objective: it may replace the incumbent only with an equal- or
 * lower-cardinality cover.
 */
static bool pool_prepend_solution(
    int **pool_solutions,
    int *pool_count,
    int pool_limit,
    const int *solution,
    int solution_len
) {
    if (
        !pool_solutions ||
        !pool_count ||
        pool_limit <= 0 ||
        !solution ||
        solution_len <= 0
    ) {
        return false;
    }

    int *normalized = (int *)malloc((size_t)solution_len * sizeof(int));
    if (!normalized) return false;
    memcpy(
        normalized,
        solution,
        (size_t)solution_len * sizeof(int)
    );

    for (int i = 1; i < solution_len; ++i) {
        int value = normalized[i];
        int j = i - 1;
        while (j >= 0 && normalized[j] > value) {
            normalized[j + 1] = normalized[j];
            --j;
        }
        normalized[j + 1] = value;
    }

    for (int p = 0; p < *pool_count; ++p) {
        if (
            memcmp(
                pool_solutions[p],
                normalized,
                (size_t)solution_len * sizeof(int)
            ) == 0
        ) {
            free(normalized);
            if (p > 0) {
                int *existing = pool_solutions[p];
                memmove(
                    &pool_solutions[1],
                    &pool_solutions[0],
                    (size_t)p * sizeof(int *)
                );
                pool_solutions[0] = existing;
            }
            return true;
        }
    }

    int retained = *pool_count;
    if (retained >= pool_limit) {
        retained = pool_limit - 1;
        free(pool_solutions[retained]);
        pool_solutions[retained] = NULL;
    } else {
        (*pool_count)++;
    }

    if (retained > 0) {
        memmove(
            &pool_solutions[1],
            &pool_solutions[0],
            (size_t)retained * sizeof(int *)
        );
    }
    pool_solutions[0] = normalized;
    return true;
}

typedef struct {
    int k;
    int ninputs;
    int noutputs;
    int *nofvalues;
    int *bit_index;
    int *word_index;
    uint64_t *shifted_mask;
    int implicant_words;
    PIstorage *PInfo;
    ThreadBuffer **buffer;
    ccubes_mutex *output_locks;
    PICoverageIndex *coverage_indices;
    int *max_shared;
    int increase;
    int *multiplier;
    bool retain_all_prime_geometries;
    double time_limit_sec;
    double checkpoint_deadline_sec;
    struct timespec start_time;
    uint64_t first_task;
    atomic_bool *time_up;
    atomic_bool *first_task_done;
    atomic_bool *failed;
    uint64_t *worker_next_tasks;
} PIWorkerContext;

typedef struct {
    int level;
    bool complete_chart;
    int ninputs;
    int noutputs;
    int *bit_index;
    int *word_index;
    uint64_t *shifted_mask;
    int implicant_words;
    PIstorage *PInfo;
    BoundedMMCSStats *stats;
    BoundedMMCSResult *results;
    double *elapsed_seconds;
    uint64_t node_limit;
    atomic_bool *fallback;
    atomic_bool *failed;
} MMCSWorkerContext;

static void pi_search_range_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    PIWorkerContext *ctx = (PIWorkerContext *)data;
    uint64_t task = start;

    while (task < end) {
        if (atomic_load_explicit(ctx->time_up, memory_order_acquire)) {
            break;
        }

        /*
         * A support task can be expensive on wide multi-output data, and a
         * low-k level may contain fewer than the former 1024-task polling
         * interval. Check before every checkpointable projection task; the
         * remaining overrun is bounded only by the duration of one in-flight
         * support task once workers have reached a safe checkpoint frontier.
         * A periodic save waits for the first pending task to finish so
         * very short intervals cannot repeatedly save without progress.
         */
        if (ctx->checkpoint_deadline_sec > 0.0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed =
                (now.tv_sec - ctx->start_time.tv_sec) +
                (now.tv_nsec - ctx->start_time.tv_nsec) / 1e9;

            if (
                elapsed >= ctx->checkpoint_deadline_sec &&
                (
                    (ctx->time_limit_sec > 0.0 && elapsed >= ctx->time_limit_sec) ||
                    atomic_load_explicit(ctx->first_task_done, memory_order_acquire)
                )
            ) {
                atomic_store_explicit(ctx->time_up, true, memory_order_release);
                break;
            }
        }

        int error = process_task(
            task,
            ctx->k,
            ctx->ninputs,
            ctx->noutputs,
            ctx->nofvalues,
            ctx->bit_index,
            ctx->word_index,
            ctx->shifted_mask,
            ctx->implicant_words,
            ctx->PInfo,
            ctx->buffer,
            worker_id,
            ctx->output_locks,
            ctx->coverage_indices,
            ctx->max_shared,
            ctx->increase,
            ctx->multiplier,
            ctx->retain_all_prime_geometries
        );

        if (error) {
            fprintf(
                stderr,
                "Error: process_task failed for task %llu at k=%d\n",
                (unsigned long long)task,
                ctx->k
            );
            atomic_store_explicit(ctx->failed, true, memory_order_release);
            atomic_store_explicit(ctx->time_up, true, memory_order_release);
            break;
        }

        if (ctx->checkpoint_deadline_sec > 0.0 && task == ctx->first_task) {
            atomic_store_explicit(ctx->first_task_done, true, memory_order_release);
        }
        task = end - task <= stride ? end : task + stride;
    }
    ctx->worker_next_tasks[worker_id] = task;
}

static void mmcs_search_range_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    MMCSWorkerContext *ctx = (MMCSWorkerContext *)data;

    for (uint64_t task = start; task < end; task += stride) {
        if (
            atomic_load_explicit(ctx->failed, memory_order_acquire) ||
            (
                ctx->node_limit > 0 &&
                atomic_load_explicit(ctx->fallback, memory_order_acquire)
            )
        ) {
            break;
        }

        int output = (int)task;
        if (
            output < 0 ||
            output >= ctx->noutputs ||
            ctx->PInfo[output].stop_search
        ) {
            continue;
        }

        struct timespec task_start;
        struct timespec task_end;
        clock_gettime(CLOCK_MONOTONIC, &task_start);
        BoundedMMCSResult result = ctx->complete_chart
            ? bounded_mmcs_generate_output_all_limited(
                &ctx->PInfo[output],
                ctx->ninputs,
                ctx->word_index,
                ctx->bit_index,
                ctx->shifted_mask,
                ctx->implicant_words,
                ctx->node_limit,
                &ctx->stats[output]
            )
            : bounded_mmcs_generate_output_level_limited(
                &ctx->PInfo[output],
                ctx->ninputs,
                ctx->level,
                ctx->word_index,
                ctx->bit_index,
                ctx->shifted_mask,
                ctx->implicant_words,
                ctx->node_limit,
                &ctx->stats[output]
            );
        clock_gettime(CLOCK_MONOTONIC, &task_end);
        if (ctx->elapsed_seconds) {
            ctx->elapsed_seconds[output] =
                (task_end.tv_sec - task_start.tv_sec) +
                (task_end.tv_nsec - task_start.tv_nsec) / 1e9;
        }
        ctx->results[output] = result;

        if (result == BOUNDED_MMCS_ERROR) {
            fprintf(
                stderr,
                "Error: bounded MMCS generation failed for output %d at k=%d\n",
                output + 1,
                ctx->level
            );
            atomic_store_explicit(ctx->failed, true, memory_order_release);
            break;
        }
        if (result == BOUNDED_MMCS_LIMIT_REACHED) {
            atomic_store_explicit(ctx->fallback, true, memory_order_release);
            break;
        }
    }
}

/*
 * Smallest blocker edge for one output: the least Hamming distance between any
 * ON row and any OFF row.  mmcs_search() pivots on the smallest uncovered edge
 * and branches once per allowed vertex in it, so this is the branching factor
 * at the root and a lower bound on it thereafter.  Computed once per run.
 */
static int min_blocker_edge(const PIstorage *pi, int ninputs) {
    int best = INT_MAX;
    for (int a = 0; a < pi->ON_minterms; ++a) {
        const int *on_row = &pi->ON_set[(size_t)a * (size_t)ninputs];
        for (int b = 0; b < pi->OFF_minterms; ++b) {
            const int *off_row = &pi->OFF_set[(size_t)b * (size_t)ninputs];
            int d = 0;
            for (int i = 0; i < ninputs && d < best; ++i) {
                if (on_row[i] != off_row[i]) ++d;
            }
            if (d > 0 && d < best) best = d;
        }
    }
    return best == INT_MAX ? 1 : best;
}

/*
 * Screen: is a branching factor of `b` to depth `k` plausibly within `budget`?
 * The tree is at most b^k, so this rejects the hopeless cases before any node
 * is spent.  It deliberately over-estimates -- the pivot edge shrinks as the
 * search proceeds and canonical forbidding trims branches -- so it is used
 * only to reject, with the budget making the finer decision.
 */
static bool mmcs_branching_plausible(int b, int k, uint64_t budget) {
    if (b <= 1) return true;
    uint64_t estimate = 1;
    for (int i = 0; i < k; ++i) {
        if (estimate > budget / (uint64_t)b) return false;
        estimate *= (uint64_t)b;
    }
    return true;
}

/*
 * Automatic routing between the ordinary CCubes pipeline and native
 * progressive IPC.
 *
 * `docs/state-overview.md` §2.1 measures the ordinary path faster at 20, 30
 * and 40 inputs on the frozen random ten-output corpus, and native IPC faster,
 * far smaller in memory, and lower in cardinality at 50 inputs. Nothing was
 * measured between those strata, and §8.3 records any claim that the 50-input
 * crossover is universal as a stop-list entry. The router therefore changes
 * behaviour only where a measurement supports the change, retains the
 * established default everywhere else, and reports every decision together
 * with the features it was made from.
 *
 * An explicit user request always wins: `--progressive-cover`, `--route=`, or
 * any option the progressive path does not accept suppresses the automatic
 * choice, and the suppression is named in the telemetry line.
 */
typedef enum {
    CCUBES_ROUTE_AUTO = 0,
    CCUBES_ROUTE_ORDINARY,
    CCUBES_ROUTE_PROGRESSIVE
} CCubesRouteMode;

/*
 * Measured. At 50 inputs IPC wins wall time (128.370 s against 155.312 s),
 * memory (21.23 MiB against a 1.79 GiB median), and merged product rows (502
 * against 532); the 100-input Fišer family is IPC-only in production.
 */
#define CCUBES_ROUTE_MIN_INPUTS 50

/*
 * Not a measured crossover. Every output of the random corpus carries at most
 * 76 ON minterms, so this guard never fires there, and the Fišer outputs that
 * do exceed it (2,768 to 18,094 ON rows) are already routed by input width.
 * It guards a shape the corpus does not contain — few inputs, many care rows —
 * against the ordinary path's projection workload, and it can only route
 * toward IPC, never away from it.
 */
#define CCUBES_ROUTE_MIN_ON_MINTERMS 10000

#define CCUBES_ROUTE_CALIBRATION "random-20-50-inputs+fiser-100"

typedef struct {
    int inputs;
    int outputs;
    int max_on_minterms;
    long long total_on_minterms;
} CCubesRouteFeatures;

static bool parse_route_mode(const char *text, CCubesRouteMode *mode) {
    if (!text || !*text || !mode) return false;
    if (strcmp(text, "auto") == 0) {
        *mode = CCUBES_ROUTE_AUTO;
        return true;
    }
    if (strcmp(text, "e0") == 0) {
        *mode = CCUBES_ROUTE_ORDINARY;
        return true;
    }
    if (strcmp(text, "ipc") == 0) {
        *mode = CCUBES_ROUTE_PROGRESSIVE;
        return true;
    }
    return false;
}

static void route_collect_features(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs,
    CCubesRouteFeatures *features
) {
    features->inputs = ninputs;
    features->outputs = noutputs;
    features->max_on_minterms = 0;
    features->total_on_minterms = 0;
    if (!PInfo) return;
    for (int o = 0; o < noutputs; ++o) {
        int on_minterms = PInfo[o].ON_minterms;
        if (on_minterms > features->max_on_minterms) {
            features->max_on_minterms = on_minterms;
        }
        features->total_on_minterms += (long long)on_minterms;
    }
}

/*
 * The decision is total: every file reaches exactly one branch. ON minterms
 * are counted per output and reduced by maximum, because the path is chosen
 * once for the whole run and the hardest output drives its cost.
 */
static bool route_auto_selects_progressive(
    const CCubesRouteFeatures *features,
    const char **reason
) {
    if (features->inputs >= CCUBES_ROUTE_MIN_INPUTS) {
        *reason = "inputs-at-or-above-crossover";
        return true;
    }
    if (features->max_on_minterms >= CCUBES_ROUTE_MIN_ON_MINTERMS) {
        *reason = "on-minterms-above-projection-guard";
        return true;
    }
    /*
     * 41 to 49 inputs was never measured. Keeping the established default
     * there means the router cannot regress an unmeasured region, and the
     * distinct reason makes those runs identifiable in logs when the region
     * is eventually calibrated.
     */
    *reason = features->inputs > 40
        ? "inputs-below-crossover-unmeasured"
        : "inputs-below-crossover";
    return false;
}

static void route_report(
    const char *decision,
    const char *source,
    const char *reason,
    const CCubesRouteFeatures *features
) {
    fprintf(
        stderr,
        "CCUBES_ROUTE decision=%s source=%s reason=%s inputs=%d outputs=%d "
        "max_on_minterms=%d total_on_minterms=%lld min_inputs=%d "
        "min_on_minterms=%d calibration=%s\n",
        decision,
        source,
        reason,
        features->inputs,
        features->outputs,
        features->max_on_minterms,
        features->total_on_minterms,
        CCUBES_ROUTE_MIN_INPUTS,
        CCUBES_ROUTE_MIN_ON_MINTERMS,
        CCUBES_ROUTE_CALIBRATION
    );
}

/* Keep logging independent of route/solver selection. On resume, wait for
 * saved source/destination paths before checking for accidental file aliases. */
static bool initialize_run_logging(const CliOptions *options, const char *source, const char *destination) {
    if (options->log_file) {
        char *ordinary_output = source ? prefix_basename(source, "ccubes_") : NULL;
        char *ipc_output = source ? prefix_basename(source, "progressive_") : NULL;
        char *checkpoint = source ? prefix_basename(source, "chk_") : NULL;
        const char *paths[] = {
            source, destination, options->resume, options->inspect,
            options->checkpoint, options->debug_file, ordinary_output, ipc_output, checkpoint,
            ccubes_option(OPTION_PROGRESSIVE_COLUMN_POOL_IN),
            ccubes_option(OPTION_PROGRESSIVE_COLUMN_POOL_OUT)
        };
        bool ok = run_log_init(options->log_file, paths, sizeof(paths) / sizeof(paths[0]));
        free(ordinary_output); free(ipc_output); free(checkpoint);
        if (!ok) return false;
    }
    if (options->debug || options->log_file) {
        int level = options->log_file && !options->debug_level_explicit
            ? DBG_INFO : options->debug_level;
        debug_init(options->debug_file, level);
    }
    return true;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        cli_help(stderr);
        return 1;
    }

    CliOptions options;
    int parse_result = cli_parse(argc, argv, &options, stderr);
    if (parse_result > 0) { cli_help(stdout); return 0; }
    if (parse_result < 0) return 1;

    const char *ipc_time_limit = ccubes_option(OPTION_PROGRESSIVE_TIME_LIMIT_SECONDS);
    const char *ipc_terminal_reserve = ccubes_option(OPTION_PROGRESSIVE_TERMINAL_RESERVE_SECONDS);
    const char *ipc_terminal_patience = ccubes_option(OPTION_PROGRESSIVE_TERMINAL_PATIENCE_SECONDS);
    const double progressive_time_limit_seconds =
        ipc_time_limit ? strtod(ipc_time_limit, NULL) : 0.0;
    const double progressive_terminal_reserve_seconds =
        ipc_terminal_reserve ? strtod(ipc_terminal_reserve, NULL) : 0.0;
    const double progressive_terminal_patience_seconds =
        ipc_terminal_patience
            ? strtod(ipc_terminal_patience, NULL)
            : CCUBES_TERMINAL_PATIENCE_DEFAULT;
    if (progressive_terminal_reserve_seconds > 0.0 &&
        progressive_terminal_reserve_seconds >= progressive_time_limit_seconds) {
        fprintf(
            stderr,
            "Error: --ipc-terminal-reserve requires --ipc-time-limit greater "
            "than the reserve.\n"
        );
        return 1;
    }

    if ((!options.resume || !options.log_file || options.inspect) &&
        !initialize_run_logging(&options, options.source, options.destination)) return 1;
    if (options.legacy_options) fprintf(stderr, "Warning: short options are deprecated; see --help for long names.\n");

    bool gurobi_ok = false;

    // Record start time for execution timing
    struct timespec start, end, startk, endk, startg, endg, startpi, endpi;
    double execution_time;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // defaults
    int search_level = 1;
    const int STOP_AFTER_EQUALITY = 1;
    int BITS_PER_WORD = options.bits_per_word;
    int THREADS = options.threads;
    bool THREADS_FORCED = THREADS > 0;
    int WEIGHT_PIC = options.weights;
    int SCP_TYPE = options.solver;
    int HYBRID_EFFORT_LEVEL = options.effort;
    int POOL_MAX = options.pool ? CCUBES_POOL_STORAGE_CAPACITY : 1;
    /*
     * Current-level coverage deduplication is always deferred until the level
     * is complete, keeping the worker-side index immutable and lock-free.
     * This flag controls only whether records are canonically sorted before
     * the common pruning/finalization pass.
     */
    bool DETERMINISTIC_PI_ORDER = options.deterministic;
    const bool CERTIFIED_MODE = options.certify;
    bool REPORT_BLOCKING_DIAGNOSTIC = options.blocking_diagnostic;
    bool PROGRESSIVE_COVER = options.progressive_cover;
    CCubesRouteMode ROUTE_MODE = CCUBES_ROUTE_ORDINARY;
    bool ROUTE_MODE_FROM_FLAG = options.route != NULL;
    if (options.route) parse_route_mode(options.route, &ROUTE_MODE);
    int PROGRESSIVE_LITERAL_POLISH_MODE = options.literal_polish;
    bool PROGRESSIVE_LITERAL_POLISH_MODE_EXPLICIT = options.literal_polish_explicit;
    char *SRC_FILE = options.source;
    char *DST_FILE = options.destination;
    // resume timing bases
    double BASE_ELAPSED = 0.0;
    double BASE_SCP = 0.0;

    // checkpoint/time limit
    double TIME_LIMIT_SEC = options.time_limit;
    double CHECKPOINT_INTERVAL_SEC = options.checkpoint_every;
    bool checkpointing = TIME_LIMIT_SEC > 0.0 || CHECKPOINT_INTERVAL_SEC > 0.0;
    char *CHK_SAVE_PATH = options.checkpoint;
    char *RESUME_PATH = options.resume;
    char *INFO_PATH = options.inspect;
    uint64_t resume_task = 0;

    int info_level = options.inspect_level;

    const char *route_source = ROUTE_MODE_FROM_FLAG ? "flag" : "default";
    if (PROGRESSIVE_COVER) {
        if (ROUTE_MODE_FROM_FLAG && ROUTE_MODE == CCUBES_ROUTE_ORDINARY) {
            fprintf(
                stderr,
                "Error: --progressive-cover requests the progressive path "
                "and --route=e0 forbids it.\n"
            );
            return 1;
        }
        ROUTE_MODE = CCUBES_ROUTE_PROGRESSIVE;
        if (!ROUTE_MODE_FROM_FLAG) route_source = "flag";
    } else if (ROUTE_MODE == CCUBES_ROUTE_PROGRESSIVE) {
        PROGRESSIVE_COVER = true;
    }

    #ifdef HAVE_GUROBI
        if (SCP_TYPE == 1) {
            gurobi_ok = gurobi_license_is_valid();
        }
    #endif

    /*
     * Routing eligibility must read the solver the user asked for, not the one
     * the fallback below leaves behind: a `-s1` run whose license is missing
     * has already been redirected once and must not be redirected again.
     */
    const bool route_gurobi_solver_requested = SCP_TYPE == 1;

    if (SCP_TYPE == 1) {
        if (!gurobi_ok) {
            SCP_TYPE = 0;
            fprintf(stderr, "Gurobi not available, falling back to the bundled hybrid solver.\n");
        }
    }

    if (POOL_MAX > 1) {
        WEIGHT_PIC = 2;
    }

    // Inspect checkpoint and exit
    if (INFO_PATH) {
        int result = print_info(INFO_PATH, info_level);
        debug_close();
        return result;
    }

    if (!SRC_FILE && !RESUME_PATH) {
        fprintf(stderr, "Error: source .pla file is required.\n");
        cli_help(stderr);
        return 1;
    }

    if (CERTIFIED_MODE && RESUME_PATH) {
        fprintf(
            stderr,
            "Error: complete-prime certification does not support checkpoint resume.\n"
        );
        return 1;
    }

    if (PROGRESSIVE_COVER && (RESUME_PATH || checkpointing)) {
        fprintf(
            stderr,
            "Error: --progressive-cover does not yet support checkpoint "
            "resume, time-limit checkpoints or periodic checkpoints.\n"
        );
        return 1;
    }

    if (
        PROGRESSIVE_COVER &&
        (
            CERTIFIED_MODE ||
            REPORT_BLOCKING_DIAGNOSTIC ||
            POOL_MAX > 1 ||
            WEIGHT_PIC != 1 ||
            SCP_TYPE != 0 ||
            HYBRID_EFFORT_LEVEL != 0
        )
    ) {
        fprintf(
            stderr,
            "Error: the progressive path currently uses the unit-cost pricing "
            "path; do not combine --progressive-cover or --route=ipc with "
            "certification, blocking diagnostics, pooling, or nondefault ordinary solver settings.\n"
        );
        return 1;
    }

    /*
     * Automatic routing may only fire on an invocation the progressive path
     * would have accepted anyway. The conditions below mirror, one for one,
     * every rule that rejects an explicit progressive request above, so the
     * router can never carry a run past a check that an explicit request
     * would have failed. Each is also a user request in its own right, which
     * is why the reason is reported rather than silently absorbed.
     */
    const char *route_block_reason = NULL;
    if (RESUME_PATH) {
        route_block_reason = "checkpoint-resume";
    } else if (TIME_LIMIT_SEC > 0.0) {
        route_block_reason = "time-limit-checkpoint";
    } else if (CHECKPOINT_INTERVAL_SEC > 0.0) {
        route_block_reason = "periodic-checkpoint";
    } else if (CERTIFIED_MODE) {
        route_block_reason = "certified-mode";
    } else if (HYBRID_EFFORT_LEVEL != 0) {
        route_block_reason = "hybrid-effort-level";
    } else if (REPORT_BLOCKING_DIAGNOSTIC) {
        route_block_reason = "blocking-diagnostic";
    } else if (POOL_MAX > 1) {
        route_block_reason = "cover-pooling";
    } else if (WEIGHT_PIC != 1) {
        route_block_reason = "prime-implicant-weighting";
    } else if (route_gurobi_solver_requested) {
        route_block_reason = "gurobi-solver";
    }

    if (CERTIFIED_MODE && checkpointing) {
        fprintf(
            stderr,
            "Error: complete-prime certification does not support "
            "time-limit or periodic checkpoints.\n"
        );
        return 1;
    }

    int max_shared = 0;
    int multiplier = 0;

    DBG_INFO_BLOCK {
        fprintf(debug_out, "--- START ---\n");
        fprintf(debug_out, "SCP_TYPE: %d\n", SCP_TYPE);
        fprintf(debug_out, "HYBRID_EFFORT_LEVEL: %d\n", HYBRID_EFFORT_LEVEL);
    }

    PIstorage *PInfo = NULL;
    int *nofvalues = NULL;
    int ninputs = 0, noutputs = 0, max_value = 0;
    int *chk_stop_counter = NULL; // resume: stop counters per output
    // Keep loaded bit parameters accessible outside the resume block
    int chk_value_bit_width_saved = 0;
    int chk_implicant_words_saved = 0;

    // Determine value bit width and implicant words
    int value_bit_width = 1; // default minimum
    int implicant_words = 0;

    // If resuming, load checkpoint before reading PLA
    if (RESUME_PATH) {

        int chk_bits = 0, chk_value_bit_width = 0, chk_implicant_words = 0;
        int chk_MAX_LEVELS=0, chk_WEIGHT_PIC=0, chk_SCP_TYPE=0, chk_POOL_MAX=0;
        char *chk_src_path = NULL;
        char *chk_dst_path = NULL;
        double chk_elapsed_total = 0.0, chk_elapsed_scp = 0.0;

        if (
            load_checkpoint(
                RESUME_PATH,
                &PInfo,
                &ninputs,
                &noutputs,
                &chk_bits,
                &chk_value_bit_width,
                &chk_implicant_words,
                &search_level,
                &chk_stop_counter,
                &chk_MAX_LEVELS,
                &chk_WEIGHT_PIC,
                &chk_SCP_TYPE,
                &chk_POOL_MAX,
                &nofvalues,
                &chk_src_path,
                &chk_dst_path,
                &chk_elapsed_total,
                &chk_elapsed_scp,
                &resume_task,
                NULL
            ) != 0
        ) {
            fprintf(stderr, "Error: failed to load checkpoint from %s\n", RESUME_PATH);
            return 1;
        }

        // If not overridden on CLI, adopt saved paths
        if (!SRC_FILE && chk_src_path) {
            SRC_FILE = chk_src_path; /* keep for later */
        } else {
            if (chk_src_path) free(chk_src_path);
        }

        if (!DST_FILE && chk_dst_path) {
            DST_FILE = chk_dst_path;
        } else {
            if (chk_dst_path) free(chk_dst_path);
        }

        if (options.log_file && !initialize_run_logging(&options, SRC_FILE, DST_FILE)) {
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            return 1;
        }

        (void)chk_MAX_LEVELS;
        WEIGHT_PIC = chk_WEIGHT_PIC;
        SCP_TYPE = chk_SCP_TYPE;
        POOL_MAX = chk_POOL_MAX;
        chk_value_bit_width_saved = chk_value_bit_width;
        chk_implicant_words_saved = chk_implicant_words;

        // Resume timing bases
        BASE_ELAPSED = chk_elapsed_total;
        BASE_SCP = chk_elapsed_scp;

        // Loaded widths are kept for later use
	    // value_bit_width and implicant_words are set after computing max_value in non-resume mode
	    // chk_stop_counter is applied after stop_counter allocation (not to be freed here)
	    // No need to store in temporary globals or static locals; assignment occurs later
	    // Variables are placed at file scope under names that do not conflict with others
	    // Values are kept in placeholders for later assignment through duplicated code blocks
        // (actual assignment happens after PLA read)
	    // Since value_bit_width is not yet declared, values are stored in local variables within this block
	    // These local values are used immediately when determining widths
	    // Dedicated flags are declared and set outside this block to support this process

        // Use the saved widths from the checkpoint
        value_bit_width = chk_value_bit_width_saved;
        implicant_words = chk_implicant_words_saved;
        BITS_PER_WORD = chk_bits;

    } else { // not resuming, normal run

        read_pla_file(
            SRC_FILE,
            &PInfo,
            &ninputs,
            &noutputs,
            &nofvalues,
            &max_value
        );

        if (PInfo == NULL || ninputs <= 0 || noutputs <= 0) {
            fprintf(stderr, "Error: Invalid .pla file or no inputs/outputs found.\n");
            return 1;
        }

        // Compute from input domain: values are encoded as 0..max_value, inclusive
        int bits_needed = (int)ceil(log2((double)(max_value + 1)));
        if (bits_needed < 1) bits_needed = 1;
        while (value_bit_width < bits_needed) {
            value_bit_width *= 2; // Round up to next power of two
        }
        if (value_bit_width > BITS_PER_WORD) {
            BITS_PER_WORD = value_bit_width; // Adjust the bits per word
        }
        implicant_words = (ninputs * value_bit_width + BITS_PER_WORD - 1) / BITS_PER_WORD;
    }

    CertifiedStopState *certified_states = NULL;
    bool adaptive_stopping_supported = certified_model_supported(
        PInfo,
        ninputs,
        noutputs
    );
    bool heuristic_pattern_supported = heuristic_pattern_model_supported(
        PInfo,
        ninputs,
        noutputs
    );
    const bool fast_hybrid_heuristic =
        adaptive_stopping_supported &&
        SCP_TYPE == 0 &&
        HYBRID_EFFORT_LEVEL == 0 &&
        !CERTIFIED_MODE;
    const bool adaptive_warning_enabled =
        adaptive_stopping_supported &&
        !fast_hybrid_heuristic &&
        !CERTIFIED_MODE;

    /*
     * The routing decision. This is the earliest point at which the file's
     * features are known and the latest at which the path can still change:
     * `PROGRESSIVE_COVER` is first consumed immediately below. The telemetry
     * line is unconditional so that every run records which path it took, on
     * what evidence, and under which thresholds — a routed run must remain
     * reproducible from its own log.
     */
    CCubesRouteFeatures route_features;
    route_collect_features(PInfo, ninputs, noutputs, &route_features);
    if (ROUTE_MODE != CCUBES_ROUTE_AUTO) {
        route_report(
            PROGRESSIVE_COVER ? "ipc" : "e0",
            route_source,
            ROUTE_MODE_FROM_FLAG || PROGRESSIVE_COVER ? "requested" : "ordinary-default",
            &route_features
        );
    } else if (route_block_reason) {
        route_report("e0", "ineligible", route_block_reason, &route_features);
    } else if (!adaptive_stopping_supported) {
        /*
         * Every progressive measurement in `docs/` was taken on nonconflicting,
         * fully specified binary point rows. Anything else keeps the ordinary
         * path, which accepts pattern rows.
         */
        route_report(
            "e0",
            "ineligible",
            "model-not-fully-specified-binary-point",
            &route_features
        );
    } else {
        const char *route_reason = NULL;
        const bool route_progressive = route_auto_selects_progressive(
            &route_features,
            &route_reason
        );
        if (route_progressive) PROGRESSIVE_COVER = true;
        route_report(
            route_progressive ? "ipc" : "e0",
            "auto",
            route_reason,
            &route_features
        );
    }

    if (!PROGRESSIVE_COVER && (ccubes_options_require_ipc() ||
        PROGRESSIVE_LITERAL_POLISH_MODE_EXPLICIT)) {
        fprintf(stderr, "Error: IPC settings require the IPC route; use --route ipc.\n");
        free(nofvalues);
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    if (!heuristic_pattern_supported && !PROGRESSIVE_COVER) {
        fprintf(
            stderr,
            "Error: CCubes requires binary point or pattern rows.\n"
        );
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    if (!adaptive_stopping_supported && CERTIFIED_MODE) {
        fprintf(
            stderr,
            "Error: --certify certified stopping requires nonconflicting, fully "
            "specified binary point rows.\n"
        );
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    if (PROGRESSIVE_COVER) {
        int progressive_max_horizon = ninputs;
        /*
         * One unchanged extension is too weak: a cover can remain flat for
         * one support level and improve at the next. Two is still explicitly
         * heuristic.
         */
        int progressive_stable_horizons = 2;
        int progressive_memory_mb = 0;
        int progressive_profile = 0;
        int progressive_seed_greedy = 1;
        int progressive_seed_memory_percent = 10;

        ProgressiveCoverMasterBackend progressive_master_backend = PROGRESSIVE_MASTER_NATIVE;
        int progressive_native_effort = 0;

        int progressive_native_repair_attempts = 16;
        bool progressive_native_repair = true;
        double progressive_native_repair_penalty = 1.0;
        double progressive_native_repair_time_share = 0.05;
        double progressive_native_feasibility_jump_seconds =
            1.0;
        bool progressive_native_feasibility_jump_work_budget =
            true;
        int progressive_native_reduce_expand =
            1;
        int progressive_native_reduce_expand_variants = 8;
        int progressive_native_reduce_expand_start_round = 8;
        int progressive_native_reduce_expand_interval = 4;
        int progressive_native_private_micro =
            1;

        int progressive_greedy_anchor_percent = 10;
        int progressive_greedy_forced_prefix_depth = -1;
        int progressive_greedy_candidate_budget = 16384;
        int progressive_greedy_max_support = 0;
        int progressive_greedy_scout_anchor_rounds = 0;
        int progressive_greedy_multi_start = 1;
        double progressive_greedy_depth2_min_relative_room = 0.24;
        int progressive_greedy_mip_interval = 8;
        double progressive_greedy_mip_seconds = 5.0;
        double progressive_greedy_final_mip_seconds = 60.0;
        int progressive_greedy_final_mip_focus = 0;
        double progressive_greedy_final_mip_best_obj_stop = -1.0;
        int progressive_mmcs_cover_bound = 1;
        int progressive_mmcs_edge_cache = 1;
        int progressive_root_gate = 1;

        double progressive_global_bound_seconds = 0.0;
        int progressive_global_bound_active_rows = 512;
        int progressive_global_bound_rounds = 5;

        /* Public arguments are validated by the CLI before any files open.
         * Algorithm tuning follows the established production policy above. */
        const char *value = ccubes_option(OPTION_PROGRESSIVE_MEMORY_MB);
        if (value) progressive_memory_mb = atoi(value);
        value = ccubes_option(OPTION_PROGRESSIVE_PROFILE);
        if (value) progressive_profile = atoi(value);
        value = ccubes_option(OPTION_PROGRESSIVE_NATIVE_EFFORT);
        if (value) progressive_native_effort = atoi(value);
        value = ccubes_option(OPTION_PROGRESSIVE_GLOBAL_BOUND_SECONDS);
        if (value) progressive_global_bound_seconds = strtod(value, NULL);
        const char *column_pool_input_path =
            ccubes_option(OPTION_PROGRESSIVE_COLUMN_POOL_IN);
        const char *column_pool_output_path =
            ccubes_option(OPTION_PROGRESSIVE_COLUMN_POOL_OUT);
        if ((column_pool_input_path || column_pool_output_path) &&
            noutputs != 1) {
            fprintf(stderr, "Error: IPC pool files require a single output.\n");
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            return 1;
        }
        const int requested_threads_raw = THREADS_FORCED ? THREADS : 0;
        int available_threads = ccubes_default_thread_count();
        if (THREADS_FORCED) {
            if (THREADS < 1) THREADS = 1;
            if (THREADS > available_threads) THREADS = available_threads;
        } else {
            THREADS = available_threads;
        }
        THREADS = ccubes_effective_thread_count(THREADS);
        ccubes_set_thread_count(THREADS);

        for (int output = 0; output < noutputs; ++output) {
            PInfo[output].cov_bits = 64;
            PInfo[output].pichart_words =
                (
                    PInfo[output].ON_minterms +
                    PInfo[output].cov_bits - 1
                ) / PInfo[output].cov_bits;
        }
        if (!DST_FILE) {
            DST_FILE = prefix_basename(SRC_FILE, "progressive_");
        }

        if (!DST_FILE) {
            fprintf(
                stderr,
                "Error: failed to determine destination .pla filename.\n"
            );
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            debug_close();
            return 1;
        }

        ProgressiveCoverOptions progressive_options;
        progressive_cover_options_default(
            &progressive_options,
            ninputs,
            progressive_max_horizon,
            progressive_stable_horizons
        );
        progressive_options.requested_threads = THREADS;
        if (progressive_memory_mb > 0) {
            progressive_options.memory_budget_bytes =
                (uint64_t)progressive_memory_mb * UINT64_C(1048576);
        }
        progressive_options.profile_pricing = progressive_profile == 1;
        progressive_options.greedy_seed = progressive_seed_greedy == 1;

        progressive_options.master_backend =
            progressive_master_backend;
        progressive_options.native_effort_level =
            progressive_native_effort;

        progressive_options.native_repair_attempt_limit =
            progressive_native_repair_attempts;
        progressive_options.native_repair_pricing =
            progressive_native_repair;
        progressive_options.native_repair_penalty =
            progressive_native_repair_penalty;
        progressive_options.native_repair_time_share =
            progressive_native_repair_time_share;
        progressive_options.native_feasibility_jump_seconds =
            progressive_native_feasibility_jump_seconds;
        progressive_options.native_feasibility_jump_work_budget =
            progressive_native_feasibility_jump_work_budget;
        progressive_options.native_reduce_expand =
            progressive_native_reduce_expand == 1;
        progressive_options.native_reduce_expand_variants =
            progressive_native_reduce_expand_variants;
        progressive_options.native_reduce_expand_start_round =
            progressive_native_reduce_expand_start_round;
        progressive_options.native_reduce_expand_interval =
            progressive_native_reduce_expand_interval;
        progressive_options.native_private_micro =
            progressive_native_private_micro == 1;

        progressive_options.greedy_anchor_percent =
            (unsigned int)progressive_greedy_anchor_percent;
        progressive_options.greedy_forced_prefix_depth =
            progressive_greedy_forced_prefix_depth;
        progressive_options.greedy_candidate_budget =
            progressive_greedy_candidate_budget;
        progressive_options.greedy_max_support =
            progressive_greedy_max_support;
        progressive_options.greedy_scout_anchor_rounds =
            progressive_greedy_scout_anchor_rounds;
        progressive_options.greedy_multi_start =
            progressive_greedy_multi_start == 1;
        progressive_options.greedy_multistart_depth2_min_relative_room =
            progressive_greedy_depth2_min_relative_room;
        progressive_options.greedy_mip_interval =
            progressive_greedy_mip_interval;
        progressive_options.greedy_mip_time_limit_seconds =
            progressive_greedy_mip_seconds;
        progressive_options.greedy_final_mip_time_limit_seconds =
            progressive_greedy_final_mip_seconds;
        progressive_options.greedy_final_mip_focus =
            progressive_greedy_final_mip_focus;
        progressive_options.greedy_final_mip_best_obj_stop =
            progressive_greedy_final_mip_best_obj_stop;
        progressive_options.disable_mmcs_cover_bound =
            progressive_mmcs_cover_bound == 0;
        progressive_options.cache_mmcs_edge_words =
            progressive_mmcs_edge_cache == 1;
        progressive_options.enable_root_gate =
            progressive_root_gate == 1;

        progressive_options.seed_memory_percent =
            (unsigned int)progressive_seed_memory_percent;
        progressive_options.time_limit_seconds =
            progressive_time_limit_seconds;
        progressive_options.native_terminal_reserve_seconds =
            progressive_terminal_reserve_seconds;
        progressive_options.native_terminal_patience_seconds =
            progressive_terminal_patience_seconds;
        progressive_options.global_bound_time_limit_seconds =
            progressive_global_bound_seconds;
        progressive_options.global_bound_active_rows =
            progressive_global_bound_active_rows;
        progressive_options.global_bound_rounds =
            progressive_global_bound_rounds;
        progressive_options.literal_polish_mode =
            PROGRESSIVE_LITERAL_POLISH_MODE;
        progressive_options.literal_polish_mode_explicit =
            PROGRESSIVE_LITERAL_POLISH_MODE_EXPLICIT;

        progressive_options.column_pool_input_path =
            column_pool_input_path;
        progressive_options.column_pool_output_path =
            column_pool_output_path;

        ProgressiveIncumbentFileContext incumbent_file_context = {
            .output_path = DST_FILE
        };
        if (noutputs == 1) {
            progressive_options.incumbent_callback =
                persist_progressive_incumbent;
            progressive_options.incumbent_callback_data =
                &incumbent_file_context;
        }
        char progressive_prefix_depth_label[16];
        if (progressive_options.greedy_forced_prefix_depth < 0) {
            strcpy(progressive_prefix_depth_label, "auto");
        } else {
            snprintf(
                progressive_prefix_depth_label,
                sizeof(progressive_prefix_depth_label),
                "%d",
                progressive_options.greedy_forced_prefix_depth
            );
        }

        fprintf(
            stderr,
            (
                    progressive_options.greedy_max_support > 0
                        ? "Notice: using IPC width-bounded greedy pricing; "
                          "nonconstant outputs are minimized independently and "
                          "pricing closure is unknown.\n"
                        : "Notice: using IPC greedy pricing; "
                          "nonconstant outputs are minimized independently and "
                          "pricing closure is unknown.\n"
                )
        );
        fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_START max_horizon=%d stable_horizons=%d outputs=%d requested_threads=%d "
                "requested_threads_raw=%d available_threads=%d configured_threads=%d backend=%s master=%s "
                "native_effort=%d repair_attempts=%d repair_pricing=%s repair_penalty=%.3f "
                "repair_time_share=%.3f feasibility_jump_mode=%s feasibility_jump_enabled=%s "
                "feasibility_jump_wall_seconds=%.3f rwls_logical_families=%d reduce_expand=%s "
                "reduce_expand_variants=%d reduce_expand_start_round=%d reduce_expand_interval=%d "
                "private_micro=%s unlimited_tail=%s memory_budget=%llu batch_share=%u worker_share=%u "
                "seed_share=%u seed_mode=%s seed_greedy=%s support_bound=%s pricing_mode=%s "
                "greedy_anchor_percent=%u greedy_prefix_depth=%s greedy_candidate_budget=%d "
                "greedy_max_support=%d greedy_scout_anchor_rounds=%d greedy_multi_start=%s "
                "greedy_depth2_min_relative_room=%.3f greedy_mip_interval=%d greedy_mip_seconds=%.3f "
                "greedy_final_mip_seconds=%.3f greedy_final_mip_focus=%d "
                "greedy_final_mip_best_obj_stop=%.3f root_gate=%s mmcs_backend=%s mmcs_cover_bound=%s "
                "mmcs_edge_cache=%s profile=%s time_limit_seconds=%.6f global_bound_seconds=%.6f "
                "global_bound_active_rows=%d global_bound_rounds=%d column_pool_in=%s column_pool_out=%s "
                "log_file=%s persistence=%s\n",
                progressive_max_horizon,
                progressive_stable_horizons,
                noutputs,
                THREADS,
                requested_threads_raw,
                available_threads,
                THREADS,
                ccubes_thread_backend(),
                progressive_options.master_backend ==
                    PROGRESSIVE_MASTER_NATIVE
                ? "native"
                : "gurobi",
                progressive_options.native_effort_level,
                progressive_options.native_repair_attempt_limit,
                progressive_options.native_repair_pricing
                ? "on"
                : "off",
                progressive_options.native_repair_penalty,
                progressive_options.native_repair_time_share,
                progressive_options.native_feasibility_jump_work_budget
                ? "work"
                : "wall",
                progressive_options.native_feasibility_jump_seconds > 0.0
                ? "yes"
                : "no",
                progressive_options.native_feasibility_jump_work_budget
                ? 0.0
                : progressive_options.native_feasibility_jump_seconds,
                progressive_options.native_feasibility_jump_logical_families,
                progressive_options.native_reduce_expand ? "on" : "off",
                progressive_options.native_reduce_expand_variants,
                progressive_options.native_reduce_expand_start_round,
                progressive_options.native_reduce_expand_interval,
                progressive_options.native_private_micro ? "on" : "off",
                progressive_options.unlimited_tail_growth ? "growth" : "plateau",
                (unsigned long long)progressive_options.memory_budget_bytes,
                progressive_options.batch_memory_percent,
                progressive_options.worker_memory_percent,
                progressive_options.seed_memory_percent,
                "master",
                progressive_options.greedy_seed ? "on" : "off",
                (
                    "diagnostic"
                ),
                "greedy",
                progressive_options.greedy_anchor_percent,
                progressive_prefix_depth_label,
                progressive_options.greedy_candidate_budget,
                progressive_options.greedy_max_support,
                progressive_options.greedy_scout_anchor_rounds,
                progressive_options.greedy_multi_start ? "on" : "off",
                progressive_options.
                greedy_multistart_depth2_min_relative_room,
                progressive_options.greedy_mip_interval,
                progressive_options.greedy_mip_time_limit_seconds,
                progressive_options.greedy_final_mip_time_limit_seconds,
                progressive_options.greedy_final_mip_focus,
                progressive_options.greedy_final_mip_best_obj_stop,
                progressive_options.enable_root_gate ? "on" : "off",
                "prepared",
                progressive_options.disable_mmcs_cover_bound
                ? "off"
                : "on",
                progressive_options.cache_mmcs_edge_words
                ? "on"
                : "off",
                progressive_options.profile_pricing ? "on" : "off",
                progressive_options.time_limit_seconds,
                progressive_options.global_bound_time_limit_seconds,
                progressive_options.global_bound_active_rows,
                progressive_options.global_bound_rounds,
                progressive_options.column_pool_input_path
                ? progressive_options.column_pool_input_path
                : "none",
                progressive_options.column_pool_output_path
                ? progressive_options.column_pool_output_path
                : "none",
                run_log_path
                ? run_log_path
                : "none",
                noutputs == 1 ? "atomic-per-incumbent" : "final-only"
            );

        ProgressiveCoverResult *progressive_results =
            (ProgressiveCoverResult *)calloc(
                (size_t)noutputs,
                sizeof(ProgressiveCoverResult)
            );
        if (!progressive_results) {
            fprintf(stderr, "Error: progressive result allocation failed.\n");
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            debug_close();
            return 1;
        }
        ProgressiveCoverStatus progressive_status =
            progressive_cover_run_multi(
            PInfo,
            ninputs,
            noutputs,
            &progressive_options,
            progressive_results
        );
        bool progressive_usable =
            progressive_status == PROGRESSIVE_COVER_OK;
        if (progressive_status == PROGRESSIVE_COVER_RESOURCE_LIMIT) {
            progressive_usable = true;
            for (int output = 0; output < noutputs; ++output) {
                if (!progressive_results[output].incumbent_valid) {
                    progressive_usable = false;
                    break;
                }
            }
        }
        if (!progressive_usable) {
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_RESULT status=%s\n",
                progressive_cover_status_name(progressive_status)
            );
            free(progressive_results);
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            debug_close();
            return 1;
        }

        long long total_connections = 0;
        double progressive_search_seconds = 0.0;
        for (int output = 0; output < noutputs; ++output) {
            ProgressiveCoverResult *progressive_result =
                &progressive_results[output];
            PInfo[output].solmin = progressive_result->cardinality;
            PInfo[output].solution = progressive_result->solution;
            progressive_result->solution = NULL;
            total_connections += progressive_result->cardinality;
            progressive_search_seconds +=
                progressive_result->elapsed_seconds;
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_OUTPUT output=%d status=%s cover=%d selected_horizon=%d "
                "final_horizon=%d stop=%s stability_start=%d stability_deferred=%d rows_on=%d/%d "
                "rows_off=%d/%d duplicates_on=%d duplicates_off=%d seed_cover=%d seed_literals=%d "
                "seed_greedy=%d seed_fallback=%d seed_complete=%s seed_master=%s seed_support=%d/%.1f/%d "
                "seed_seconds=%.6f seed_bytes=%llu support_bound=%d bound_anchors=%d/%.1f/%d "
                "bound_skipped=%d bound_skip_applied=%s bound_seconds=%.6f bound_bytes=%llu pricing_mode=%s "
                "partial_anchor_limit=%d greedy_anchor_limit=%d greedy_prefix_depth=%d "
                "greedy_candidate_budget=%d greedy_max_support=%d greedy_multistart=%d/%d "
                "greedy_secondary=%d greedy_multistart_seconds=%.6f greedy_rounds=%d greedy_mip_interval=%d "
                "greedy_mip_seconds=%.3f greedy_final_mip_seconds=%.3f greedy_mip_calls=%d/%d "
                "greedy_mip_skipped=%d native_repair=%d/%d repair_skipped_optimal=%d repair_sideways=%d "
                "repair_rejected=%d repair_no_improvement=%d/%d repair_destroyed=%llu "
                "repair_uncovered=%llu/%d repair_residual_columns=%llu/%d repair_lagrangian_complete=%d/%d "
                "repair_lagrangian_columns=%llu repair_fallbacks=%d repair_fallback_columns=%llu "
                "repair_pruned=%llu repair_seconds=%.6f/%.6f repair_budget_seconds=%.6f "
                "repair_budget_skips=%d repair_deadline_stops=%d dual_calls=%d dual_seconds=%.6f "
                "time_grace_seconds=%.3f time_extensions=%d time_extension_seconds=%.6f "
                "time_descending_at_base=%s time_base_cover=%d greedy_support=%d/%.1f/%d "
                "greedy_anchors=%llu greedy_constructed=%llu greedy_improving=%llu greedy_duplicates=%llu "
                "greedy_literal_evaluations=%llu greedy_seconds=%.6f batch_capacity=%d batch_bytes=%llu "
                "pricing_threads=%d worker_bytes=%llu peak_incidence=%llu columns=%d rounds=%d "
                "oracle_calls=%d oracle_nodes=%llu lp=%.9f lp_scope=%s mip_bound=%.6f mip_gap=%.6f "
                "pool_loaded=%d pool_saved=%d pool_load_seconds=%.6f pool_save_seconds=%.6f "
                "restricted_master_optimal=%s incumbent=%s incumbent_generation=%d pricing_closed=%s "
                "closed_horizon=%d search_seconds=%.6f\n",
                output + 1,
                progressive_result->resource_limited
                    ? "resource-limit"
                    : "ok",
                progressive_result->cardinality,
                progressive_result->selected_horizon,
                progressive_result->final_horizon,
                progressive_cover_stop_reason_name(
                    progressive_result->stop_reason
                ),
                progressive_result->stability_start_horizon,
                progressive_result->stability_deferred_horizons,
                progressive_result->on_rows_unique,
                progressive_result->on_rows_original,
                progressive_result->off_rows_unique,
                progressive_result->off_rows_original,
                progressive_result->on_duplicates,
                progressive_result->off_duplicates,
                progressive_result->seed_cardinality,
                progressive_result->seed_literal_count,
                progressive_result->seed_greedy_cubes,
                progressive_result->seed_fallback_cubes,
                progressive_result->seed_greedy_complete ? "yes" : "no",
                progressive_result->seed_master_inserted ? "yes" : "no",
                progressive_result->seed_support_minimum,
                progressive_result->seed_support_median,
                progressive_result->seed_support_maximum,
                (double)progressive_result->seed_elapsed_nanoseconds / 1e9,
                (unsigned long long)
                    progressive_result->seed_allocated_bytes,
                progressive_result->support_lower_bound,
                progressive_result->support_bound_anchor_minimum,
                progressive_result->support_bound_anchor_median,
                progressive_result->support_bound_anchor_maximum,
                progressive_result->horizons_skipped,
                progressive_result->support_bound_skip_applied
                    ? "yes"
                    : "no",
                (double)
                    progressive_result->support_bound_elapsed_nanoseconds /
                    1e9,
                (unsigned long long)
                    progressive_result->support_bound_allocated_bytes,
                "greedy",
                progressive_result->partial_anchor_limit,
                progressive_result->greedy_anchor_limit,
                progressive_result->greedy_forced_prefix_depth,
                progressive_result->greedy_candidate_budget,
                progressive_options.greedy_max_support,
                progressive_result->greedy_multistart_improvements,
                progressive_result->greedy_multistart_attempts,
                progressive_result->greedy_secondary_cardinality,
                (double)
                    progressive_result->
                        greedy_multistart_elapsed_nanoseconds /
                    1e9,
                progressive_result->greedy_rounds,
                progressive_options.greedy_mip_interval,
                progressive_result->greedy_mip_time_limit_seconds,
                progressive_result->greedy_final_mip_time_limit_seconds,
                progressive_result->greedy_intermediate_mip_calls,
                progressive_result->greedy_final_mip_calls,
                progressive_result->greedy_mip_rounds_skipped,
                progressive_result->native_repair_improvements,
                progressive_result->native_repair_attempts,
                progressive_result->native_repair_skipped_optimal,
                progressive_result->native_repair_sideways_moves,
                progressive_result->native_repair_rejections,
                progressive_result->
                    native_repair_attempts_since_improvement,
                progressive_result->
                    native_repair_max_attempts_without_improvement,
                (unsigned long long)
                    progressive_result->native_repair_destroyed_columns,
                (unsigned long long)
                    progressive_result->native_repair_uncovered_rows,
                progressive_result->native_repair_max_uncovered_rows,
                (unsigned long long)
                    progressive_result->native_repair_residual_columns,
                progressive_result->
                    native_repair_max_residual_columns,
                progressive_result->
                    native_repair_lagrangian_completions,
                progressive_result->
                    native_repair_lagrangian_attempts,
                (unsigned long long)
                    progressive_result->native_repair_lagrangian_columns,
                progressive_result->native_repair_greedy_fallbacks,
                (unsigned long long)
                    progressive_result->native_repair_greedy_columns,
                (unsigned long long)
                    progressive_result->native_repair_redundant_removed,
                (double)
                    progressive_result->
                        native_repair_prepare_nanoseconds /
                    1e9,
                (double)
                    progressive_result->
                        native_repair_solve_nanoseconds /
                    1e9,
                (double)
                    progressive_result->
                        native_repair_budget_nanoseconds /
                    1e9,
                progressive_result->native_repair_budget_skips,
                progressive_result->native_repair_deadline_stops,
                progressive_result->native_dual_calls,
                (double)progressive_result->native_dual_nanoseconds / 1e9,
                progressive_result->native_soft_target_grace_seconds,
                progressive_result->native_soft_target_extensions,
                (double)
                    progressive_result->
                        native_soft_target_extension_nanoseconds /
                    1e9,
                progressive_result->native_soft_target_descending_at_base
                    ? "yes"
                    : "no",
                progressive_result->native_soft_target_base_cardinality,
                progressive_result->greedy_support_minimum,
                progressive_result->greedy_support_mean,
                progressive_result->greedy_support_maximum,
                (unsigned long long)
                    progressive_result->greedy_anchors_attempted,
                (unsigned long long)
                    progressive_result->greedy_candidates_constructed,
                (unsigned long long)
                    progressive_result->greedy_improving_candidates,
                (unsigned long long)
                    progressive_result->greedy_duplicate_candidates,
                (unsigned long long)
                    progressive_result->greedy_literal_evaluations,
                (double)
                    progressive_result->greedy_elapsed_nanoseconds /
                    1e9,
                progressive_result->batch_capacity,
                (unsigned long long)
                    progressive_result->batch_allocated_bytes,
                progressive_result->pricing_threads,
                (unsigned long long)
                    progressive_result->estimated_bytes_per_worker,
                (unsigned long long)
                    progressive_result->peak_incidence_entries,
                progressive_result->retained_columns,
                progressive_result->pricing_rounds,
                progressive_result->oracle_calls,
                (unsigned long long)
                    progressive_result->oracle_search_nodes,
                progressive_result->lp_objective,
                progressive_result->pricing_closed
                    ? "global"
                    : "restricted_master",
                progressive_result->integer_best_bound,
                progressive_result->integer_mip_gap,
                progressive_result->column_pool_loaded_columns,
                progressive_result->column_pool_saved_columns,
                (double)
                    progressive_result->column_pool_load_nanoseconds /
                    1e9,
                (double)
                    progressive_result->column_pool_save_nanoseconds /
                    1e9,
                progressive_result->restricted_master_optimal
                    ? "yes"
                    : "no",
                progressive_cover_incumbent_source_name(
                    progressive_result->incumbent_source
                ),
                progressive_result->incumbent_generation,
                progressive_result->pricing_closed ? "yes" : "no",
                progressive_result->pricing_closed_horizon,
                progressive_result->elapsed_seconds
            );
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_WORK output=%d "
                "work_thread_backend=%s work_threads_forced=%s "
                "work_threads_requested_raw=%d "
                "work_threads_available=%d work_threads_configured=%d "
                "work_threads_pricing=%d work_threads_rwls=%d "
                "work_pricing_rounds=%d "
                "work_column_growth_rounds=%d "
                "work_pricing_oracle_calls=%d "
                "work_pricing_oracle_nodes=%llu "
                "work_pricing_anchors=%llu "
                "work_pricing_candidates=%llu "
                "work_pricing_improving=%llu "
                "work_pricing_literal_evaluations=%llu "
                "work_retained_columns=%llu "
                "work_dual_calls=%d work_dual_iterations=%llu "
                "work_rwls_runs=%d work_rwls_families=%llu "
                "work_rwls_logical_states=%d "
                "work_rwls_steps=%llu work_rwls_add_moves=%llu "
                "work_rwls_remove_moves=%llu "
                "work_rwls_fixed_steps=%llu work_rwls_soft_steps=%llu "
                "work_rwls_epochs=%llu "
                "work_rwls_productive_epoch_extensions=%llu "
                "work_rwls_budget_mode=%s "
                "work_rwls_shared_cache=%llu "
                "work_rwls_cache_source=%s "
                "work_rwls_packed_threshold=%llu "
                "work_rwls_dense_incidence_invocations=%d "
                "work_rwls_packed_incidence_invocations=%d "
                "work_phase_checkpoints=%d "
                "work_phase_promotions=%d "
                "work_phase_grace_promotions=%d "
                "work_phase_budget_promotions=%d "
                "work_phase_growth_promotions=%d "
                "work_phase_pool_checkpoints=%d "
                "work_phase_stopped=%s "
                "work_reduce_expand_passes=%d "
                "work_reduce_expand_variants=%d "
                "work_reduce_expand_inserted=%d "
                "work_repair_attempts=%d work_repair_improvements=%d "
                "work_repair_destroyed=%llu "
                "work_repair_uncovered=%llu "
                "work_repair_residual_columns=%llu "
                "work_repair_prepare_seconds=%.6f "
                "work_repair_solve_seconds=%.6f "
                "work_repair_budget_seconds=%.6f "
                "work_repair_budget_skips=%d "
                "work_repair_deadline_stops=%d\n",
                output + 1,
                ccubes_thread_backend(),
                THREADS_FORCED ? "yes" : "no",
                requested_threads_raw,
                available_threads,
                THREADS,
                progressive_result->pricing_threads,
                progressive_result->native_feasibility_jump_threads_used,
                progressive_result->greedy_rounds,
                progressive_result->pricing_rounds,
                progressive_result->oracle_calls,
                (unsigned long long)
                    progressive_result->oracle_search_nodes,
                (unsigned long long)
                    progressive_result->greedy_anchors_attempted,
                (unsigned long long)
                    progressive_result->greedy_candidates_constructed,
                (unsigned long long)
                    progressive_result->greedy_improving_candidates,
                (unsigned long long)
                    progressive_result->greedy_literal_evaluations,
                (unsigned long long)
                    progressive_result->work_retained_columns,
                progressive_result->native_dual_calls,
                (unsigned long long)
                    progressive_result->native_dual_iterations,
                progressive_result->native_feasibility_jump_runs,
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_restart_families,
                progressive_result->
                    native_feasibility_jump_logical_family_states,
                (unsigned long long)
                    progressive_result->native_feasibility_jump_iterations,
                (unsigned long long)
                    progressive_result->native_feasibility_jump_add_moves,
                (unsigned long long)
                    progressive_result->native_feasibility_jump_remove_moves,
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_fixed_iterations,
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_soft_iterations,
                (unsigned long long)
                    progressive_result->native_feasibility_jump_work_epochs,
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_productive_epoch_extensions,
                progressive_options.native_feasibility_jump_work_budget
                    ? "work"
                    : "wall",
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_shared_cache_bytes,
                progressive_rwls_shared_cache_source_name(
                    (ProgressiveRWLSSharedCacheSource)
                        progressive_result->
                            native_feasibility_jump_shared_cache_source
                ),
                (unsigned long long)
                    progressive_result->
                        native_feasibility_jump_packed_threshold_bytes,
                progressive_result->
                    native_feasibility_jump_dense_incidence_invocations,
                progressive_result->
                    native_feasibility_jump_packed_incidence_invocations,
                progressive_result->native_adaptive_phase_checkpoints,
                progressive_result->native_adaptive_phase_promotions,
                progressive_result->native_adaptive_phase_grace_promotions,
                progressive_result->native_adaptive_phase_budget_promotions,
                progressive_result->native_adaptive_phase_growth_promotions,
                progressive_result->native_adaptive_phase_pool_checkpoints,
                progressive_result->native_adaptive_phase_stopped
                    ? "yes"
                    : "no",
                progressive_result->native_reduce_expand_passes,
                progressive_result->native_reduce_expand_variants,
                progressive_result->native_reduce_expand_inserted_columns,
                progressive_result->native_repair_attempts,
                progressive_result->native_repair_improvements,
                (unsigned long long)
                    progressive_result->native_repair_destroyed_columns,
                (unsigned long long)
                    progressive_result->native_repair_uncovered_rows,
                (unsigned long long)
                    progressive_result->native_repair_residual_columns,
                (double)
                    progressive_result->native_repair_prepare_nanoseconds /
                    1e9,
                (double)
                    progressive_result->native_repair_solve_nanoseconds /
                    1e9,
                (double)
                    progressive_result->native_repair_budget_nanoseconds /
                    1e9,
                progressive_result->native_repair_budget_skips,
                progressive_result->native_repair_deadline_stops
            );

        }
        if (!run_log_check_path(DST_FILE)) {
            free(progressive_results);
            free(nofvalues);
            free(chk_stop_counter);
            cleanup(PInfo, NULL);
            debug_close();
            return 1;
        }
        write_pla_file(DST_FILE, PInfo);
        long long merged_products = written_pla_product_count(DST_FILE);

        clock_gettime(CLOCK_MONOTONIC, &end);
        execution_time =
            (end.tv_sec - start.tv_sec) +
            (end.tv_nsec - start.tv_nsec) / 1e9;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_RESULT status=%s outputs=%d "
            "products=%lld connections=%lld search_seconds=%.6f "
            "output=%s\n",
            progressive_cover_status_name(progressive_status),
            noutputs,
            merged_products,
            total_connections,
            progressive_search_seconds,
            DST_FILE
        );
        fprintf(
            stderr,
            "Execution completed in %.3f seconds\n",
            execution_time
        );
        fprintf(
            stderr,
            "CCUBES_TIMING_TOTAL pi_generation=0.000000 "
            "scp=%.6f other=%.6f total=%.6f\n",
            progressive_search_seconds,
            execution_time - progressive_search_seconds,
            execution_time
        );

        for (int output = 0; output < noutputs; ++output) {
            progressive_cover_result_destroy(&progressive_results[output]);
        }
        free(progressive_results);
        free(nofvalues);
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        debug_close();
        return 0;
    }

    /*
     * Build OFF compatibility masks before the shared projection layout. The
     * layout can then omit OFF rows for masked outputs: those rows are tested
     * by bitset intersection and never need a mixed-radix projection code.
     * Checkpoints omit both derived structures, so resume rebuilds them here.
     */
    if (!prepare_off_compat_masks(
        PInfo,
        ninputs,
        noutputs,
        nofvalues
    )) {
        fprintf(stderr, "Error: OFF compatibility-mask allocation failed.\n");
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    /*
     * The PLA reader stores the same ON row independently in multiple output
     * partitions. Build one immutable ON-row table before starting workers so
     * each required row can be projected once across outputs. Scalar OFF rows
     * intentionally remain lazy inside process_task.
     */
    if (!prepare_shared_projection_rows(PInfo, ninputs, noutputs)) {
        fprintf(stderr, "Error: shared projection-layout allocation failed.\n");
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    DBG_INFO_BLOCK {
        size_t projection_memberships = 0u;
        int masked_outputs = 0;
        size_t off_mask_bytes = 0u;
        for (int output = 0; output < noutputs; ++output) {
            projection_memberships += (size_t)PInfo[output].ON_minterms;
            if (PInfo[output].off_compat_masks) {
                masked_outputs++;
                off_mask_bytes +=
                    (size_t)PInfo[output].off_mask_count *
                    (size_t)PInfo[output].off_mask_words *
                    sizeof(uint64_t);
                off_mask_bytes +=
                    ((size_t)ninputs + 1u) * sizeof(int);
            }
        }
        fprintf(
            debug_out,
            "Shared projection rows: %d unique across %zu output memberships\n",
            PInfo[0].projection_row_count,
            projection_memberships
        );
        fprintf(
            debug_out,
            "OFF compatibility masks: %d outputs, %zu bytes\n",
            masked_outputs,
            off_mask_bytes
        );
    }

    /*
     * Worker-side support unranking calls nchoosek repeatedly. Build the
     * read-only Pascal table before any PI-generation threads can start.
     */
    if (!nchoosek_prepare(ninputs)) {
        fprintf(stderr, "Error: binomial lookup-table allocation failed.\n");
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    if (CERTIFIED_MODE) {
        fprintf(stderr, "Notice: --certify uses complete-prime MMCS certification.\n");
    }

    if (!adaptive_stopping_supported) {
        fprintf(
            stderr,
            "Notice: input-dash pattern rows detected; using heuristic plateau stopping. "
            "Adaptive pair diagnosis and certified stopping are unavailable.\n"
        );
        if (REPORT_BLOCKING_DIAGNOSTIC) {
            fprintf(
                stderr,
                "Notice: --blocking-diagnostic is ignored because the blocking diagnostic requires point rows.\n"
            );
            REPORT_BLOCKING_DIAGNOSTIC = false;
        }
    } else if (fast_hybrid_heuristic) {
        fprintf(
            stderr,
            "Notice: hybrid effort 0 uses fast heuristic plateau stopping; "
            "automatic certification is disabled.\n"
        );
    }

    certified_states = calloc((size_t)noutputs, sizeof(CertifiedStopState));
    if (!certified_states) {
        fprintf(stderr, "Error: stopping-state allocation failed.\n");
        free(chk_stop_counter);
        cleanup(PInfo, NULL);
        return 1;
    }

    for (int o = 0; o < noutputs; ++o) {
        certified_stop_state_reset(&certified_states[o]);
    }
    uint64_t VALUE_BIT_MASK = (1ULL << value_bit_width) - 1ULL;

    // Ensure pichart_words are aligned with the coverage packing width
    // (already set on load for resume)
    const int COV_BITS_PER_WORD = coverage_bits_per_word(BITS_PER_WORD);
    PInfo[0].pichart_words = (PInfo[0].ON_minterms + COV_BITS_PER_WORD - 1) / COV_BITS_PER_WORD;
    PInfo[0].cov_bits = COV_BITS_PER_WORD;
    for (int o = 0; o < noutputs; o++) {
        PInfo[o].pichart_words = (PInfo[o].ON_minterms + COV_BITS_PER_WORD - 1) / COV_BITS_PER_WORD;
        PInfo[o].cov_bits = COV_BITS_PER_WORD;
    }

    uint64_t initial_pi_bytes = 0;
    int active_output_count = 0;
    int constant_zero_count = 0;
    int constant_one_count = 0;
    const int initial_pi_capacity = ordinary_initial_pi_capacity(
        PInfo,
        noutputs,
        implicant_words,
        &initial_pi_bytes,
        &active_output_count,
        &constant_zero_count,
        &constant_one_count
    );
    const int increase = CCUBES_PI_CAPACITY_GROWTH;
    fprintf(
        stderr,
        "CCUBES_PI_CAPACITY initial=%d growth=%d "
        "estimated_initial_bytes=%llu budget_bytes=%llu "
        "active_outputs=%d constant_zero=%d constant_one=%d\n",
        initial_pi_capacity,
        increase,
        (unsigned long long)initial_pi_bytes,
        (unsigned long long)CCUBES_INITIAL_PI_MEMORY_BYTES,
        active_output_count,
        constant_zero_count,
        constant_one_count
    );

    // many PIs will have the same coverage, but they don't necessarily cover the same minterms
    // to employ row dominance when solving the coverage matrix, we need to compare the coverage of
    // the current PI with the coverage of the previous PIs. If this PI survives the comparison, its
    // coverage has to be added in the "covered" vector, at the last index of the PI coverage with
    // the same number of (covered) minterms
    // int last_index[ON_minterms]; // descending order

    int available_threads = ccubes_default_thread_count();
    if (THREADS_FORCED) {
        if (THREADS < 1) THREADS = 1;
        if (THREADS > available_threads) THREADS = available_threads;
    } else {
        THREADS = available_threads;
    }
    THREADS = ccubes_effective_thread_count(THREADS);
    ccubes_set_thread_count(THREADS);

    /* opt-in (--lock-stats) profiling of the PI-insertion locks */
    if (!lock_stats_init(options.lock_stats, THREADS, noutputs)) {
        fprintf(stderr, "Error: lock statistics allocation failed\n");
        return 1;
    }

    // Implement thread-private buffer space
    ThreadBuffer **buffer = (ThreadBuffer**)calloc(THREADS, sizeof(ThreadBuffer*));
    if (!buffer) {
        fprintf(stderr, "Error: buffer allocation failed\n");
        cleanup(PInfo, buffer);
        return 1;
    }

    int stop_counter[noutputs];
    BlockingStopState blocking_stop_states[noutputs];
    // Initialize per-output state (allocate only if not resuming)
    for (int o = 0; o < noutputs; o++) {
        certified_blocking_state_init(&blocking_stop_states[o]);

        // --- temporary debugging code ---
        // printf("Output %d, ON_minterms: %d, OFF_minterms: %d\n", o + 1, PInfo[o].ON_minterms, PInfo[o].OFF_minterms);
        // --------------------------------

        // the original ON-set was transposed to a column-major matrix
        int ON_minterms = PInfo[o].ON_minterms;
        const bool constant_zero = ON_minterms <= 0;
        const bool constant_one =
            ON_minterms > 0 && PInfo[o].OFF_minterms <= 0;
        const bool constant_output = constant_zero || constant_one;

        /*
         * Retained PI arrays grow transactionally.  A fixed 500,000-column
         * reservation per output made the virtual chart scale with
         * outputs * ON rows before one PI had been found.  Share one bounded
         * initial memory envelope across active outputs instead; constants
         * need at most one synthetic all-dash column.
         */
        if (!RESUME_PATH) {
            PInfo[o].estimPI = constant_output
                ? 1
                : initial_pi_capacity;
        } else {
            if (PInfo[o].estimPI < PInfo[o].foundPI) {
                PInfo[o].estimPI = PInfo[o].foundPI;
            }
            if (PInfo[o].estimPI < 1) PInfo[o].estimPI = 1;
        }

        // allocate only if not already present (resume)
        if (!PInfo[o].covered) {
            PInfo[o].covered = (int *) calloc(PInfo[o].estimPI, sizeof(int));
            if (!PInfo[o].covered) {
                fprintf(stderr, "Error: Memory allocation failed for covered array\n");
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        if (ON_minterms > 0 && !PInfo[o].last_index) {
            PInfo[o].last_index = (int *)calloc(
                (size_t)ON_minterms, sizeof(int)
            );
        }
        if (ON_minterms > 0 && !PInfo[o].k_last_index) {
            PInfo[o].k_last_index = (int *)calloc(
                (size_t)ON_minterms, sizeof(int)
            );
        }
        if (
            ON_minterms > 0 &&
            (!PInfo[o].last_index || !PInfo[o].k_last_index)
        ) {
            fprintf(stderr, "Error: Memory allocation failed for index arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].pichart_words = (ON_minterms + COV_BITS_PER_WORD - 1) / COV_BITS_PER_WORD; // Words needed per coverage matrix columns
        PInfo[o].cov_bits = COV_BITS_PER_WORD;
        if (PInfo[o].pichart_words > 0 && !PInfo[o].pichart_pos) {
            PInfo[o].pichart_pos = (uint64_t *)calloc(
                (size_t)PInfo[o].estimPI *
                    (size_t)PInfo[o].pichart_words,
                sizeof(uint64_t)
            );
            if (!PInfo[o].pichart_pos) {
                fprintf(stderr, "Error: Memory allocation failed for pichart_pos\n");
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        if (!PInfo[o].implicants_pos) {
            PInfo[o].implicants_pos = (uint64_t *)calloc(
                (size_t)PInfo[o].estimPI * (size_t)implicant_words,
                sizeof(uint64_t)
            );
        }
        if (!PInfo[o].implicants_val) {
            PInfo[o].implicants_val = (uint64_t *)calloc(
                (size_t)PInfo[o].estimPI * (size_t)implicant_words,
                sizeof(uint64_t)
            );
        }
        if (!PInfo[o].implicants_pos || !PInfo[o].implicants_val) {
            fprintf(stderr, "Error: Memory allocation failed for implicants arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!PInfo[o].shared) {
            PInfo[o].shared = (int *)calloc(
                (size_t)PInfo[o].estimPI, sizeof(int)
            );
        }
        if (!PInfo[o].covsum) {
            PInfo[o].covsum = (int *)calloc(
                (size_t)PInfo[o].estimPI, sizeof(int)
            );
        }
        if (!PInfo[o].shared || !PInfo[o].covsum) {
            fprintf(stderr, "Error: Memory allocation failed for shared/covsum arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!RESUME_PATH) {
            PInfo[o].foundPI = constant_one ? 1 : 0;
            PInfo[o].solmin = constant_one ? 1 : 0;
        }

        // the previous (level k - 1), minimum number of PIs to solve the coverage matrix
        if (!RESUME_PATH) {
            PInfo[o].prevsolmin = constant_one
                ? 1
                : (constant_zero ? 0 : ON_minterms + 1);
        }

        // the positions of the PIs solving the coverage matrix
        // this vector can never be lengthier than the number of ON minterms (ON_minterms)
        if (ON_minterms > 0 && !PInfo[o].previndices) {
            PInfo[o].previndices = (int *)calloc(
                (size_t)ON_minterms, sizeof(int)
            );
        }
        if (ON_minterms > 0 && !PInfo[o].indices) {
            PInfo[o].indices = (int *)calloc(
                (size_t)ON_minterms, sizeof(int)
            );
        }
        if (
            ON_minterms > 0 &&
            (!PInfo[o].previndices || !PInfo[o].indices)
        ) {
            fprintf(stderr, "Error: Memory allocation failed for indices arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!RESUME_PATH) {
            PInfo[o].ON_set_covered = constant_output;
        }

        if (ON_minterms > 0 && !PInfo[o].cov_word_index) {
            PInfo[o].cov_word_index = (int *)calloc(
                (size_t)ON_minterms, sizeof(int)
            );
        }
        if (ON_minterms > 0 && !PInfo[o].shifted_cov_mask) {
            PInfo[o].shifted_cov_mask = (uint64_t *)calloc(
                (size_t)ON_minterms, sizeof(uint64_t)
            );
        }
        if (
            ON_minterms > 0 &&
            (!PInfo[o].cov_word_index || !PInfo[o].shifted_cov_mask)
        ) {
            fprintf(stderr, "Error: Memory allocation failed for coverage arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        for (int r = 0; r < ON_minterms; r++) {
            PInfo[o].cov_word_index[r] = r / COV_BITS_PER_WORD;
            PInfo[o].shifted_cov_mask[r] = (1ULL << (r % COV_BITS_PER_WORD));
        }
        if (!RESUME_PATH && constant_one) {
            for (int word = 0; word < PInfo[o].pichart_words; ++word) {
                int remaining = ON_minterms - word * COV_BITS_PER_WORD;
                PInfo[o].pichart_pos[word] = remaining >= 64
                    ? UINT64_MAX
                    : (UINT64_C(1) << remaining) - 1u;
            }
            PInfo[o].covsum[0] = ON_minterms;
            PInfo[o].covered[0] = ON_minterms;
            PInfo[o].indices[0] = 0;
            PInfo[o].previndices[0] = 0;
        }

        // If resuming, restore stop counter from checkpoint
        if (chk_stop_counter) {
            stop_counter[o] = chk_stop_counter[o];
        } else {
            stop_counter[o] = 0;
        }
        if (adaptive_stopping_supported && RESUME_PATH && stop_counter[o] > 0) {
            blocking_stop_states[o].reported = true;
            if (!PInfo[o].stop_search) {
                PInfo[o].stop_search = true;
                fprintf(
                    stderr,
                    "Notice: output %d resumed at a stored heuristic plateau; "
                    "use a fresh --certify run for global certification.\n",
                    o + 1
                );
            }
        }

        // store the number of PIs at each level of complexity k
        if (!PInfo[o].nofpi) {
            PInfo[o].nofpi = (int *) calloc(ninputs, sizeof(int));
            if (!PInfo[o].nofpi) {
                fprintf(stderr, "Error: Memory allocation failed for nofpi array\n");
                cleanup(PInfo, buffer);
                return 1;
            }
        }
        if (!RESUME_PATH && constant_one && ninputs > 0) {
            PInfo[o].nofpi[0] = 1;
        }

        // Preserve stop status when resuming; otherwise initialize it.
        if (!RESUME_PATH) {
            PInfo[o].stop_search = constant_output;
        }

        PInfo[o].solution = NULL;
        PInfo[o].pool_count = 0;
        PInfo[o].pool_solutions = NULL;
        if (POOL_MAX > 1) {
            PInfo[o].pool_solutions = (int **) calloc((size_t)POOL_MAX, sizeof(int*));
            if (!PInfo[o].pool_solutions) {
                fprintf(stderr, "Error: Memory allocation failed for pool_solutions\n");
                cleanup(PInfo, buffer);
                return 1;
            }
            if (constant_one) {
                PInfo[o].pool_solutions[0] = (int *)malloc(sizeof(int));
                if (!PInfo[o].pool_solutions[0]) {
                    fprintf(
                        stderr,
                        "Error: constant-one pool allocation failed\n"
                    );
                    cleanup(PInfo, buffer);
                    return 1;
                }
                PInfo[o].pool_solutions[0][0] = 0;
                PInfo[o].pool_count = 1;
            }
        }
    }

    if (chk_stop_counter) {
        free(chk_stop_counter);
        chk_stop_counter = NULL;
    }

    ccubes_mutex *output_locks = (ccubes_mutex *)calloc((size_t)noutputs, sizeof(ccubes_mutex));
    if (!output_locks) {
        fprintf(stderr, "Error: output lock allocation failed\n");
        cleanup(PInfo, buffer);
        return 1;
    }
    for (int o = 0; o < noutputs; o++) {
        if (!ccubes_mutex_init(&output_locks[o])) {
            fprintf(stderr, "Error: output lock initialization failed\n");
            destroy_output_locks(output_locks, o);
            cleanup(PInfo, buffer);
            return 1;
        }
    }

    // allocate memory buffer for each thread
    for (int t = 0; t < THREADS; t++) {
        buffer[t] = (ThreadBuffer*)calloc(noutputs, sizeof(ThreadBuffer));
        if (!buffer[t]) {
            fprintf(stderr, "Error: buffer[%d] allocation failed\n", t);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            return 1;
        }

        buffer[t]->threads = THREADS;

        for (int o = 0; o < noutputs; o++) {
            buffer[t][o].capacity       = 0;
            buffer[t][o].found          = 0;
        }
    }

    DBG_INFO_BLOCK {
        fprintf(debug_out, "Bits per word: %d\n", BITS_PER_WORD);
        fprintf(debug_out, "Implicant words: %d\n", implicant_words);
        fprintf(debug_out, "value bit width: %d\n", value_bit_width);
        // fprintf(debug_out, "value bit mask: %lld\n", VALUE_BIT_MASK);
        fprintf(debug_out, "ON-set minterms: %d\n", PInfo[0].ON_minterms);
        fprintf(debug_out, "threads: %d\n", THREADS);
        fprintf(debug_out, "thread backend: %s\n", ccubes_thread_backend());
        fprintf(debug_out, "deterministic PI order: %s\n", DETERMINISTIC_PI_ORDER ? "yes" : "no");
    }

    int bit_index[ninputs];  //
    int word_index[ninputs]; //   for the implicants matrix
    uint64_t shifted_mask[ninputs];
    for (int r = 0; r < ninputs; r++) {
        bit_index[r] = (r % (BITS_PER_WORD / value_bit_width)) * value_bit_width;   // Bit position within the word
        word_index[r] = r / (BITS_PER_WORD / value_bit_width);                      // Word index within the implicant
        shifted_mask[r] = (VALUE_BIT_MASK << bit_index[r]);
    }

    double scp_time = 0.0, k_scp_time = 0.0, pi_generation_time = 0.0;
    double k_pi_generation_time = 0.0;
    double next_periodic_checkpoint_sec = CHECKPOINT_INTERVAL_SEC;
    bool continuing_level = false;
    double certification_enumeration_wall_time = 0.0;
    double certification_enumeration_worker_time = 0.0;
    double certification_chart_solve_time = 0.0;
    uint64_t certification_prime_count = 0;

    /*
     * Widest branching factor over all outputs.  A level trial falls back as
     * soon as any single output exhausts its budget, so the level is only
     * worth attempting if the hardest output is plausible.
     */
    int mmcs_branching = 1;
    if (!CERTIFIED_MODE && adaptive_stopping_supported) {
        for (int o = 0; o < noutputs; ++o) {
            int b = min_blocker_edge(&PInfo[o], ninputs);
            if (b > mmcs_branching) mmcs_branching = b;
        }

        DBG_INFO_BLOCK {
            fprintf(
                debug_out,
                "CCUBES_MMCS_SCREEN branching=%d\n",
                mmcs_branching
            );
        }
    }

    int output_completed_level[noutputs];
    for (int o = 0; o < noutputs; ++o) {
        output_completed_level[o] = search_level - 1;
    }

    int k;
    for (k = search_level; k <= ninputs; k++) {
        const bool complete_mmcs_chart = CERTIFIED_MODE;
        int level_start[noutputs];
        bool level_active[noutputs];
        for (int o = 0; o < noutputs; ++o) {
            level_active[o] = !PInfo[o].stop_search;
            int start_pi = (k > 1 && PInfo[o].nofpi) ? PInfo[o].nofpi[k - 2] : 0;
            if (start_pi < 0 || start_pi > PInfo[o].foundPI) {
                start_pi = PInfo[o].foundPI;
            }
            level_start[o] = start_pi;
        }

        if (!continuing_level) {
            k_scp_time = 0.0;
            k_pi_generation_time = 0.0;
            clock_gettime(CLOCK_MONOTONIC, &startk);
        }

        uint64_t projection_tasks = nchoosek(ninputs, k);
        bool automatic_mmcs_trial =
            !complete_mmcs_chart &&
            adaptive_stopping_supported &&
            !RESUME_PATH &&
            !checkpointing &&
            (
                projection_tasks == 0 ||
                projection_tasks > CCUBES_AUTO_MMCS_MIN_PROJECTION_TASKS
            );
        bool attempt_mmcs = complete_mmcs_chart || automatic_mmcs_trial;
        /*
         * Break-even trial budget.  A search node costs about
         * O(ninputs * edge_count) and a projection task about
         * O((|ON| + |OFF|) * k); the row-count factor appears in both, so the
         * total work compares as
         *
         *     nodes * ninputs   vs   C(ninputs, k) * k
         *
         * Equating them gives the budget below: abandon MMCS exactly once it
         * has cost roughly what projection would, making the worst case ~2x
         * projection while leaving the win unbounded.  A fixed budget cannot
         * do this -- it is far too generous when C(n,k) is small and far too
         * mean when C(n,k) is astronomical, which is precisely when MMCS is
         * the only viable generator.  Divide before multiplying: C(n,k)
         * reaches 10^9 at n=100, k=6.
         */
        uint64_t automatic_mmcs_node_limit = 0;
        if (automatic_mmcs_trial && projection_tasks > 0) {
            uint64_t budget = projection_tasks / (uint64_t)ninputs;
            if (budget > UINT64_MAX / (uint64_t)k) {
                budget = UINT64_MAX / (uint64_t)k;
            }
            budget *= (uint64_t)k;
            if (budget < CCUBES_AUTO_MMCS_MIN_NODE_BUDGET) {
                budget = CCUBES_AUTO_MMCS_MIN_NODE_BUDGET;
            }
            if (budget > CCUBES_AUTO_MMCS_MAX_NODE_LIMIT) {
                budget = CCUBES_AUTO_MMCS_MAX_NODE_LIMIT;
            }
            /*
             * Reject up front when the branching factor cannot fit the budget.
             * Without this the break-even budget is itself the cost of being
             * wrong, and on wide instances that is paid at every level.
             */
            if (!mmcs_branching_plausible(mmcs_branching, k, budget)) {
                automatic_mmcs_trial = false;
                attempt_mmcs = complete_mmcs_chart;
            } else {
                automatic_mmcs_node_limit = budget;
            }
        }

        int mmcs_threads =
            automatic_mmcs_trial && THREADS > CCUBES_AUTO_MMCS_MAX_TRIAL_THREADS
                ? CCUBES_AUTO_MMCS_MAX_TRIAL_THREADS
                : THREADS;
        bool use_mmcs_level = false;
        uint64_t maxtasks = attempt_mmcs
            ? (uint64_t)noutputs
            : projection_tasks;
        double checkpoint_deadline_sec = TIME_LIMIT_SEC;
        if (
            next_periodic_checkpoint_sec > 0.0 &&
            (checkpoint_deadline_sec <= 0.0 ||
             next_periodic_checkpoint_sec < checkpoint_deadline_sec)
        ) {
            checkpoint_deadline_sec = next_periodic_checkpoint_sec;
        }
        PICoverageIndex *coverage_indices = NULL;
        atomic_bool time_up;
        atomic_init(&time_up, false);
        atomic_bool first_task_done;
        atomic_init(&first_task_done, false);
        atomic_bool generation_failed;
        atomic_init(&generation_failed, false);
        uint64_t worker_next_tasks[THREADS];
        for (int t = 0; t < THREADS; ++t) {
            worker_next_tasks[t] = UINT64_MAX;
        }
        uint64_t start_task = resume_task <= maxtasks ? resume_task : maxtasks;

        if (!attempt_mmcs && maxtasks == 0) {
            // overflow, too many tasks
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            // exit(EXIT_FAILURE);
            return(1);
        }

        if (!attempt_mmcs) {
            if (!build_pi_coverage_indices(
                &coverage_indices,
                PInfo,
                noutputs,
                level_start,
                implicant_words
            )) {
                fprintf(stderr, "Error: failed to build PI coverage indices\n");
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        DBG_INFO_BLOCK {
            fprintf(debug_out, "\nk: %d\n", k);
            if (attempt_mmcs) {
                fprintf(debug_out, "output tasks: %llu\n",
                    (unsigned long long)maxtasks);
                if (automatic_mmcs_trial) {
                    fprintf(
                        debug_out,
                        "automatic MMCS trial node limit per output: %llu "
                        "(threads: %d)\n",
                        (unsigned long long)automatic_mmcs_node_limit,
                        mmcs_threads
                    );
                }
            } else {
                fprintf(debug_out, "maxtasks: %llu\n",
                    (unsigned long long)maxtasks);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &startpi);
        BoundedMMCSStats mmcs_stats[noutputs];
        memset(mmcs_stats, 0, sizeof(mmcs_stats));
        BoundedMMCSResult mmcs_results[noutputs];
        double mmcs_elapsed_seconds[noutputs];
        memset(mmcs_elapsed_seconds, 0, sizeof(mmcs_elapsed_seconds));
        for (int o = 0; o < noutputs; ++o) {
            mmcs_results[o] = BOUNDED_MMCS_COMPLETE;
        }
        atomic_bool mmcs_fallback;
        atomic_init(&mmcs_fallback, false);
        atomic_bool mmcs_failed;
        atomic_init(&mmcs_failed, false);

        if (attempt_mmcs) {
            MMCSWorkerContext mmcs_ctx = {
                .level = k,
                .complete_chart = complete_mmcs_chart,
                .ninputs = ninputs,
                .noutputs = noutputs,
                .bit_index = bit_index,
                .word_index = word_index,
                .shifted_mask = shifted_mask,
                .implicant_words = implicant_words,
                .PInfo = PInfo,
                .stats = mmcs_stats,
                .results = mmcs_results,
                .elapsed_seconds = mmcs_elapsed_seconds,
                .node_limit = automatic_mmcs_node_limit,
                .fallback = &mmcs_fallback,
                .failed = &mmcs_failed
            };

            if (!ccubes_parallel_for(
                0,
                (uint64_t)noutputs,
                mmcs_threads,
                false,
                mmcs_search_range_worker,
                &mmcs_ctx
            ) || atomic_load_explicit(&mmcs_failed, memory_order_acquire)) {
                fprintf(stderr, "Error: bounded MMCS PI search failed\n");
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }

            bool fallback = atomic_load_explicit(
                &mmcs_fallback,
                memory_order_acquire
            );
            if (automatic_mmcs_trial && fallback) {
                /*
                 * A bounded MMCS level is transactional.  Completed outputs
                 * may already have appended records, while the limiting
                 * output appended none.  Reset every output to the common
                 * level boundary before projection completes the full level.
                 */
                for (int o = 0; o < noutputs; ++o) {
                    PInfo[o].foundPI = level_start[o];
                }
                attempt_mmcs = false;
                maxtasks = projection_tasks;
                if (maxtasks == 0) {
                    fprintf(
                        stderr,
                        "Error: projection task count overflow after MMCS trial fallback\n"
                    );
                    destroy_output_locks(output_locks, noutputs);
                    cleanup(PInfo, buffer);
                    return 1;
                }
                if (!build_pi_coverage_indices(
                    &coverage_indices,
                    PInfo,
                    noutputs,
                    level_start,
                    implicant_words
                )) {
                    fprintf(stderr, "Error: failed to build PI coverage indices\n");
                    destroy_output_locks(output_locks, noutputs);
                    cleanup(PInfo, buffer);
                    return 1;
                }
            } else {
                use_mmcs_level = true;
                if (!bounded_mmcs_mark_level_sharing(
                    PInfo,
                    noutputs,
                    level_start,
                    implicant_words,
                    &max_shared
                )) {
                    fprintf(stderr, "Error: failed to mark MMCS PI sharing\n");
                    destroy_output_locks(output_locks, noutputs);
                    cleanup(PInfo, buffer);
                    return 1;
                }
            }
        }

        if (!use_mmcs_level) {
            PIWorkerContext pi_ctx = {
                .k = k,
                .ninputs = ninputs,
                .noutputs = noutputs,
                .nofvalues = nofvalues,
                .bit_index = bit_index,
                .word_index = word_index,
                .shifted_mask = shifted_mask,
                .implicant_words = implicant_words,
                .PInfo = PInfo,
                .buffer = buffer,
                .output_locks = output_locks,
                .coverage_indices = coverage_indices,
                .max_shared = &max_shared,
                .increase = increase,
                .multiplier = &multiplier,
                .retain_all_prime_geometries =
                    false,
                .time_limit_sec = TIME_LIMIT_SEC,
                .checkpoint_deadline_sec = checkpoint_deadline_sec,
                .start_time = start,
                .first_task = start_task,
                .time_up = &time_up,
                .first_task_done = &first_task_done,
                .failed = &generation_failed,
                .worker_next_tasks = worker_next_tasks
            };
            if (!ccubes_parallel_for(
                start_task,
                maxtasks,
                THREADS,
                true,
                pi_search_range_worker,
                &pi_ctx
            )) {
                fprintf(stderr, "Error: failed to start workers for PI search\n");
                destroy_pi_coverage_indices(coverage_indices, noutputs);
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        if (atomic_load_explicit(&generation_failed, memory_order_acquire)) {
            destroy_pi_coverage_indices(coverage_indices, noutputs);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            debug_close();
            return 1;
        }

        // The first unfinished task of any worker is the safe resume point.
        bool level_time_up = atomic_load_explicit(&time_up, memory_order_acquire);
        resume_task = maxtasks;
        if (level_time_up) {
            for (int t = 0; t < THREADS; ++t) {
                if (worker_next_tasks[t] < resume_task) {
                    resume_task = worker_next_tasks[t];
                }
            }
        }
        DBG_INFO_BLOCK {
            uint64_t mmcs_trial_nodes = 0;
            for (int o = 0; o < noutputs; ++o) {
                mmcs_trial_nodes += mmcs_stats[o].search_nodes;
            }
            if (use_mmcs_level) {
                for (int o = 0; o < noutputs; ++o) {
                    fprintf(
                        debug_out,
                        "bounded MMCS output %d: nodes=%llu completed=%llu "
                        "unique=%llu duplicates=%llu\n",
                        o + 1,
                        (unsigned long long)mmcs_stats[o].search_nodes,
                        (unsigned long long)mmcs_stats[o].completed_transversals,
                        (unsigned long long)mmcs_stats[o].unique_cubes,
                        (unsigned long long)mmcs_stats[o].duplicate_cubes
                    );
                }
                fprintf(
                    debug_out,
                    "CCUBES_PI_GENERATOR level=%d selected=%s "
                    "trial_nodes=%llu\n",
                    k,
                    complete_mmcs_chart ? "complete-mmcs" : "mmcs",
                    (unsigned long long)mmcs_trial_nodes
                );
            } else {
                for (int o = 0; o < noutputs; ++o) {
                    uint64_t rejected = atomic_load_explicit(
                        &coverage_indices[o].subsumption_rejections,
                        memory_order_relaxed
                    );
                    if (rejected > 0) {
                        fprintf(
                            debug_out,
                            "subsumption rejections, output %d: %llu\n",
                            o + 1,
                            (unsigned long long)rejected
                        );
                    }
                }
                fprintf(
                    debug_out,
                    "CCUBES_PI_GENERATOR level=%d selected=projection "
                    "reason=%s trial_nodes=%llu projection_tasks=%llu\n",
                    k,
                    automatic_mmcs_trial
                        ? "mmcs-node-limit"
                        : adaptive_stopping_supported
                            ? "small-support-level"
                            : "unsupported-model",
                    (unsigned long long)mmcs_trial_nodes,
                    (unsigned long long)projection_tasks
                );
            }
        }
        destroy_pi_coverage_indices(coverage_indices, noutputs);

        for (int o = 0; o < noutputs; ++o) {
            if (!level_active[o]) continue;
            if (complete_mmcs_chart) {
                /*
                 * Keep every complete-chart geometry. Cross-level coverage
                 * equivalents can differ in support and secondary weight, so
                 * ordinary same-level coverage pruning is not valid here.
                 */
                certified_stop_observe_complete_prime_chart(
                    &certified_states[o]
                );
            } else if (!finalize_pi_level(
                &PInfo[o],
                implicant_words,
                level_start[o],
                DETERMINISTIC_PI_ORDER,
                POOL_MAX > 1 || WEIGHT_PIC == 2,
                false
            )) {
                fprintf(
                    stderr,
                    "Error: failed to finalize PI level for output %d\n",
                    o + 1
                );
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &endpi);
        double generation_slice_seconds =
            (endpi.tv_sec - startpi.tv_sec) +
            (endpi.tv_nsec - startpi.tv_nsec) / 1e9;
        k_pi_generation_time += generation_slice_seconds;
        pi_generation_time += generation_slice_seconds;

        if (complete_mmcs_chart) {
            certification_enumeration_wall_time += k_pi_generation_time;
            uint64_t level_primes = 0;
            double level_worker_seconds = 0.0;
            for (int o = 0; o < noutputs; ++o) {
                level_primes += (uint64_t)PInfo[o].foundPI;
                level_worker_seconds += mmcs_elapsed_seconds[o];
                fprintf(
                    stderr,
                    "CCUBES_CERTIFICATION_ENUMERATION output=%d "
                    "primes=%d nodes=%llu seconds=%.6f\n",
                    o + 1,
                    PInfo[o].foundPI,
                    (unsigned long long)mmcs_stats[o].search_nodes,
                    mmcs_elapsed_seconds[o]
                );
            }
            certification_prime_count += level_primes;
            certification_enumeration_worker_time += level_worker_seconds;
            fprintf(
                stderr,
                "CCUBES_CERTIFICATION_TIMING phase=enumeration "
                "wall_seconds=%.6f worker_seconds=%.6f outputs=%d "
                "primes=%llu\n",
                k_pi_generation_time,
                level_worker_seconds,
                noutputs,
                (unsigned long long)level_primes
            );
        }

        if (level_time_up) {
            if (!CHK_SAVE_PATH) {
                if (RESUME_PATH) {
                    CHK_SAVE_PATH = RESUME_PATH; // overwrite the same checkpoint when resuming
                } else if (SRC_FILE) {
                    CHK_SAVE_PATH = prefix_basename(SRC_FILE, "chk_");
                } else {
                    CHK_SAVE_PATH = "ccubes_checkpoint.bin";
                }
            }

            // Determine intended destination path to persist
            const char *dst_to_save = DST_FILE;
            char *tmp_dst_alloc = NULL;
            if (!dst_to_save && SRC_FILE) {
                tmp_dst_alloc = prefix_basename(SRC_FILE, "ccubes_");
                dst_to_save = tmp_dst_alloc; // may be NULL on OOM
            }
            if (!CHK_SAVE_PATH || !run_log_check_path(CHK_SAVE_PATH)) {
                if (!CHK_SAVE_PATH) fprintf(stderr, "Error: failed to allocate checkpoint path.\n");
                free(tmp_dst_alloc);
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                debug_close();
                return 1;
            }
            struct timespec checkpoint_start;
            clock_gettime(CLOCK_MONOTONIC, &checkpoint_start);
            double checkpoint_elapsed =
                (checkpoint_start.tv_sec - start.tv_sec) +
                (checkpoint_start.tv_nsec - start.tv_nsec) / 1e9;
            if (
                save_checkpoint(
                    CHK_SAVE_PATH,
                    PInfo,
                    ninputs,
                    noutputs,
                    BITS_PER_WORD,
                    value_bit_width,
                    implicant_words,
                    k,
                    stop_counter,
                    STOP_AFTER_EQUALITY,
                    WEIGHT_PIC,
                    SCP_TYPE,
                    POOL_MAX,
                    SRC_FILE,
                    dst_to_save,
                    BASE_ELAPSED + checkpoint_elapsed,
                    BASE_SCP + scp_time,
                    resume_task
                ) != 0
            ) {
                fprintf(stderr, "Error: failed to save checkpoint to %s.\n", CHK_SAVE_PATH);
                free(tmp_dst_alloc);
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                debug_close();
                return 1;
            }
            free(tmp_dst_alloc);

            struct timespec checkpoint_end;
            clock_gettime(CLOCK_MONOTONIC, &checkpoint_end);
            double elapsed_after_save =
                (checkpoint_end.tv_sec - start.tv_sec) +
                (checkpoint_end.tv_nsec - start.tv_nsec) / 1e9;
            double pct = (double)resume_task * 100.0 / (double)maxtasks;
            if (TIME_LIMIT_SEC > 0.0 && elapsed_after_save >= TIME_LIMIT_SEC) {
                fprintf(
                    stderr,
                    "Time limit reached. Checkpoint saved at %.2f%% of level k=%d.\n",
                    pct, k
                );
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                debug_close();
                return 0;
            }

            fprintf(
                stderr,
                "Periodic checkpoint saved at %.2f%% of level k=%d; continuing.\n",
                pct, k
            );
            next_periodic_checkpoint_sec = elapsed_after_save + CHECKPOINT_INTERVAL_SEC;
            continuing_level = true;
            --k;
            continue;
        }

        continuing_level = false;
        resume_task = 0;
        for (int o = 0; o < noutputs; ++o) {
            if (level_active[o]) output_completed_level[o] = k;
        }

        // solve the PI charts per output
        if (POOL_MAX == 1) {
            for (int o = 0; o < noutputs; o++) {
                int *foundPI = &PInfo[o].foundPI;
                bool *ON_set_covered = &PInfo[o].ON_set_covered;
                int ON_minterms = PInfo[o].ON_minterms;
                int *prevsolmin = &PInfo[o].prevsolmin;
                int *solmin = &PInfo[o].solmin;
                int *previndices = PInfo[o].previndices;
                int *indices = PInfo[o].indices;

                PInfo[o].nofpi[k - 1] = *foundPI; // TODO move this after checking the coverage at this level of k

                /* built here: pichart_pos may have been reallocated during
                   this level's generation, and foundPI is now final for k */
                PIChartView chart = pi_chart_view(&PInfo[o]);

                if (*foundPI > 0 && !*ON_set_covered) {
                    bool test_coverage = true;

                    int r = 0;
                    while (r < ON_minterms && test_coverage) {

                        bool minterm_covered = false;
                        int c = 0;

                        while (c < *foundPI && !minterm_covered) {
                            minterm_covered = chart_covers(&chart, c, r);
                            c++;
                        }

                        test_coverage = minterm_covered;
                        r++;
                    }

                    *ON_set_covered = test_coverage;
                }

                if (*ON_set_covered && !PInfo[o].stop_search) {
                    bool boundary_exact = SCP_TYPE == 1;
                    DBG_TRACE_BLOCK {
                        fprintf(debug_out, "Output %d, found PIs: %d", o + 1, *foundPI);
                    }

                    double *weights = complete_mmcs_chart
                        ? build_complete_cover_weights(
                            &PInfo[o],
                            *foundPI,
                            ninputs,
                            implicant_words,
                            word_index,
                            shifted_mask,
                            WEIGHT_PIC
                        )
                        : build_cover_weights(
                            &PInfo[o],
                            *foundPI,
                            k,
                            WEIGHT_PIC
                        );
                    if (WEIGHT_PIC > 0 && !weights) {
                        fprintf(stderr, "Error: Memory allocation failed for cover weights\n");
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    clock_gettime(CLOCK_MONOTONIC, &startg);

                    if (SCP_TYPE == 0) { // Bundled hybrid solver
                        const int *initial_solution =
                            *prevsolmin > 0 &&
                            *prevsolmin <= ON_minterms
                                ? previndices
                                : NULL;
                        solve_scp_lagrangian(
                            &chart,
                            weights,
                            indices,
                            solmin,
                            HYBRID_EFFORT_LEVEL,
                            initial_solution,
                            initial_solution ? *prevsolmin : 0,
                            CERTIFIED_MODE
                        );
                        boundary_exact = lagrangian_last_run_proved_optimal();
                        print_hybrid_stats(o);
                    }

                    if (SCP_TYPE == 1) { // Gurobi: blended multi-objective
                        gurobi_multiobjective(
                            &chart,
                            weights,
                            *prevsolmin > 0 &&
                                *prevsolmin <= ON_minterms
                                    ? previndices
                                    : NULL,
                            *prevsolmin > 0 &&
                                *prevsolmin <= ON_minterms
                                    ? *prevsolmin
                                    : 0,
                            indices,
                            solmin
                        );
                    }

                    bool probe_due =
                        *solmin > 0 &&
                        adaptive_stopping_supported &&
                        *prevsolmin > 1 &&
                        *prevsolmin <= ON_minterms &&
                        *solmin >= *prevsolmin &&
                        stop_counter[o] + 1 >= STOP_AFTER_EQUALITY;
                    if (probe_due) {
                        const int pre_probe_found = *foundPI;
                        const uint64_t pair_limit =
                            HYBRID_EFFORT_LEVEL >= 2
                                ? UINT64_C(1024)
                                : UINT64_C(128);
                        const int candidate_limit =
                            HYBRID_EFFORT_LEVEL >= 2 ? 128 : 32;
                        PlateauProbeStats probe_stats;

                        if (!plateau_probe_append_candidates(
                            &PInfo[o],
                            ninputs,
                            implicant_words,
                            previndices,
                            *prevsolmin,
                            bit_index,
                            word_index,
                            shifted_mask,
                            pair_limit,
                            candidate_limit,
                            &probe_stats
                        )) {
                            fprintf(
                                stderr,
                                "Error: plateau probe failed for output %d.\n",
                                o + 1
                            );
                            free(weights);
                            destroy_output_locks(output_locks, noutputs);
                            cleanup(PInfo, buffer);
                            return 1;
                        }

                        if (probe_stats.candidates_appended > 0) {
                            PInfo[o].nofpi[k - 1] = *foundPI;
                            chart = pi_chart_view(&PInfo[o]);
                            free(weights);
                            weights = complete_mmcs_chart
                                ? build_complete_cover_weights(
                                    &PInfo[o],
                                    *foundPI,
                                    ninputs,
                                    implicant_words,
                                    word_index,
                                    shifted_mask,
                                    WEIGHT_PIC
                                )
                                : build_cover_weights(
                                    &PInfo[o],
                                    *foundPI,
                                    k,
                                    WEIGHT_PIC
                                );
                            if (WEIGHT_PIC > 0 && !weights) {
                                fprintf(
                                    stderr,
                                    "Error: Memory allocation failed for "
                                    "plateau-probe cover weights\n"
                                );
                                destroy_output_locks(output_locks, noutputs);
                                cleanup(PInfo, buffer);
                                return 1;
                            }

                            if (SCP_TYPE == 0) {
                                solve_scp_lagrangian(
                                    &chart,
                                    weights,
                                    indices,
                                    solmin,
                                    HYBRID_EFFORT_LEVEL,
                                    previndices,
                                    *prevsolmin,
                                    CERTIFIED_MODE
                                );
                                boundary_exact =
                                    lagrangian_last_run_proved_optimal();
                                print_hybrid_stats(o);
                            }
                            if (SCP_TYPE == 1) {
                                gurobi_multiobjective(
                                    &chart,
                                    weights,
                                    previndices,
                                    *prevsolmin,
                                    indices,
                                    solmin
                                );
                                boundary_exact = true;
                            }

                            if (*solmin >= *prevsolmin) {
                                *foundPI = pre_probe_found;
                                PInfo[o].nofpi[k - 1] = pre_probe_found;
                            }
                        }

                        DBG_INFO_BLOCK {
                            fprintf(
                                debug_out,
                                "CCUBES_PLATEAU_PROBE level=%d output=%d "
                                "witnesses=%d pairs=%llu compatible=%llu "
                                "generated=%d appended=%d improved=%s\n",
                                k,
                                o + 1,
                                probe_stats.private_witnesses,
                                (unsigned long long)
                                    probe_stats.pairs_examined,
                                (unsigned long long)
                                    probe_stats.compatible_pairs,
                                probe_stats.candidates_generated,
                                probe_stats.candidates_appended,
                                *solmin < *prevsolmin ? "yes" : "no"
                            );
                        }
                    }

                    if (*solmin == 0) {
                        DBG_ERROR_BLOCK {
                            fprintf(debug_out, "Error: solving the minterm coverage failed.\n");
                        }
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    clock_gettime(CLOCK_MONOTONIC, &endg);

                    execution_time =
                        (endg.tv_sec - startg.tv_sec) +
                        (endg.tv_nsec - startg.tv_nsec) / 1e9;

                    k_scp_time += execution_time;

                    if (complete_mmcs_chart) {
                        certification_chart_solve_time += execution_time;
                        fprintf(
                            stderr,
                            "CCUBES_CERTIFICATION_CHART output=%d rows=%d "
                            "primes=%d cover=%d boundary_exact=%d "
                            "seconds=%.6f\n",
                            o + 1,
                            ON_minterms,
                            *foundPI,
                            *solmin,
                            boundary_exact ? 1 : 0,
                            execution_time
                        );
                    }

                    DBG_TRACE_BLOCK {
                        fprintf(debug_out, " (SCP %.3fs)", execution_time);
                    }

                    free(weights);

                    DBG_TRACE_BLOCK {
                        // // print the PI chart
                        // fprintf(debug_out, "\nPI chart for output %d(%d):\n", o + 1, ON_minterms);
                        // for (int r = 0; r < ON_minterms; r++) {
                        //     for (int c = 0; c < *foundPI; c++) {
                        //         fprintf(debug_out, "%d ", pichart[c * ON_minterms + r]);
                        //     }
                        //     fprintf(debug_out, "\n");
                        // }
                    }

                    bool plateau_triggered = false;
                    if (*solmin < *prevsolmin) {
                        // either solmin is smaller than the previously found solmin,
                        // or it is the very first time a solmin was found

                        *prevsolmin = *solmin;
                        for (int i = 0; i < *solmin; i++) {
                            previndices[i] = indices[i];
                        }

                        stop_counter[o] = 0;

                        DBG_TRACE_BLOCK {
                            if (!PInfo[o].stop_search) {
                                fprintf(debug_out, " solution (%d)", *solmin);
                                for (int i = 0; i < *solmin; i++) {
                                    fprintf(debug_out, " %d", indices[i] + 1);
                                }
                                fprintf(debug_out, "\n");
                            }
                        }
                    } else {
                        // the minimum number of PIs did not change in the current level of complexity
                        // we can safely retain the less complex PIs from the previous level
                        for (int i = 0; i < *solmin; i++) {
                            indices[i] = previndices[i];
                        }

                        *solmin = *prevsolmin;
                        stop_counter[o]++;

                        DBG_TRACE_BLOCK {
                            if (!PInfo[o].stop_search) {
                                fprintf(debug_out, " solution (%d)", *solmin);
                                for (int i = 0; i < *solmin; i++) {
                                    fprintf(debug_out, " %d", indices[i] + 1);
                                }
                                fprintf(debug_out, "%s\n", stop_counter[o] >= STOP_AFTER_EQUALITY ? " -- stopping search" : "");
                            }
                        }

                        plateau_triggered = stop_counter[o] >= STOP_AFTER_EQUALITY;
                    }

                    if (adaptive_stopping_supported && !certified_blocking_observe_plateau(
                        &blocking_stop_states[o],
                        REPORT_BLOCKING_DIAGNOSTIC,
                        adaptive_warning_enabled,
                        plateau_triggered,
                        stderr,
                        o + 1,
                        &PInfo[o],
                        ninputs,
                        k,
                        indices,
                        *solmin,
                        boundary_exact,
                        false
                    )) {
                        fprintf(stderr, "Error: blocking diagnostic failed for output %d.\n", o + 1);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }
                    PInfo[o].stop_search = certified_stop_policy_decision(
                        certified_states ? &certified_states[o] : NULL,
                        CERTIFIED_MODE,
                        plateau_triggered,
                        k,
                        *solmin,
                        boundary_exact,
                        stderr,
                        o + 1
                    );

                    DBG_TRACE_BLOCK {
                        // fprintf(debug_out, "solmin: %d%s\n", *solmin, PInfo[o].stop_search ? ", stopping search" : "");
                        // for (int i = 0; i < *solmin; i++) {
                        //     fprintf(debug_out, "%d ", indices[i] + 1);
                        // }
                        // fprintf(debug_out, "\n");

                        // // Print the PIs:
                        // for (int c = 0; c < *solmin; c++) {
                        //     for (int r = 0; r < ninputs; r++) {
                        //         int value = 0;
                        //         int position = r * (*solmin) + c;

                        //         if (PInfo[o].implicants_pos[indices[c] * implicant_words + word_index[r]] & shifted_mask[r]) {
                        //             value = 1 + (int)((PInfo[o].implicants_val[indices[c] * implicant_words + word_index[r]] >> bit_index[r]) & VALUE_BIT_MASK);
                        //         }

                        //         fprintf(debug_out, "%d ", value);
                        //     }
                        //     fprintf(debug_out, "\n");
                        // }
                    }
                }

                for (int i = 0; i < ON_minterms; i++) {
                    PInfo[o].last_index[i] = PInfo[o].k_last_index[i];
                }
            } // end of outputs loop solving SCP

        } // end of searching for solutions without pooling

        if (POOL_MAX > 1) { // pooled solutions

            double pool_execution_time[noutputs];
            bool pool_boundary_exact[noutputs];

            for (int o = 0; o < noutputs; o++) {
                pool_execution_time[o] = 0.0;
                pool_boundary_exact[o] = SCP_TYPE == 1;

                int *foundPI = &PInfo[o].foundPI;
                bool *ON_set_covered = &PInfo[o].ON_set_covered;
                int ON_minterms = PInfo[o].ON_minterms;
                int *solmin = &PInfo[o].solmin;
                int *pool_count = &PInfo[o].pool_count;
                int **pool_solutions = PInfo[o].pool_solutions;

                PInfo[o].nofpi[k - 1] = *foundPI;

                /* built here: pichart_pos may have been reallocated during
                   this level's generation, and foundPI is now final for k */
                PIChartView chart = pi_chart_view(&PInfo[o]);

                if (*foundPI > 0 && !*ON_set_covered) {
                    bool test_coverage = true;

                    int r = 0;
                    while (r < ON_minterms && test_coverage) {

                        bool minterm_covered = false;
                        int c = 0;

                        while (c < *foundPI && !minterm_covered) {
                            minterm_covered = chart_covers(&chart, c, r);
                            c++;
                        }

                        test_coverage = minterm_covered;
                        r++;
                    }

                    *ON_set_covered = test_coverage;
                }

                if (*ON_set_covered && !PInfo[o].stop_search) {
                    clear_output_solution_pool(&PInfo[o]);
                    double *weights = complete_mmcs_chart
                        ? build_complete_cover_weights(
                            &PInfo[o],
                            *foundPI,
                            ninputs,
                            implicant_words,
                            word_index,
                            shifted_mask,
                            WEIGHT_PIC
                        )
                        : build_cover_weights(
                            &PInfo[o],
                            *foundPI,
                            k,
                            WEIGHT_PIC
                        );
                    if (WEIGHT_PIC > 0 && !weights) {
                        fprintf(stderr, "Error: Memory allocation failed for cover weights\n");
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    int *indices = PInfo[o].indices;
                    int *baseline_indices = (int *)malloc(
                        (size_t)ON_minterms * sizeof(int)
                    );
                    if (!baseline_indices) {
                        fprintf(
                            stderr,
                            "Error: Memory allocation failed for pool baseline\n"
                        );
                        free(weights);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    clock_gettime(CLOCK_MONOTONIC, &startg);

                    /*
                     * Establish the ordinary sharing-aware incumbent first.
                     * This path may use column dominance because it needs only
                     * one cover. Alternative discovery remains deferred until
                     * the boundary is about to plateau.
                     */
                    if (SCP_TYPE == 0) {
                        solve_scp_lagrangian(
                            &chart,
                            weights,
                            indices,
                            solmin,
                            HYBRID_EFFORT_LEVEL,
                            PInfo[o].prevsolmin > 0 &&
                                PInfo[o].prevsolmin <= ON_minterms
                                    ? PInfo[o].previndices
                                    : NULL,
                            PInfo[o].prevsolmin > 0 &&
                                PInfo[o].prevsolmin <= ON_minterms
                                    ? PInfo[o].prevsolmin
                                    : 0,
                            CERTIFIED_MODE
                        );
                        pool_boundary_exact[o] = lagrangian_last_run_proved_optimal();
                        print_hybrid_stats(o);
                    }

                    if (SCP_TYPE == 1) {
                        gurobi_multiobjective(
                            &chart,
                            weights,
                            PInfo[o].prevsolmin > 0 &&
                                PInfo[o].prevsolmin <= ON_minterms
                                    ? PInfo[o].previndices
                                    : NULL,
                            PInfo[o].prevsolmin > 0 &&
                                PInfo[o].prevsolmin <= ON_minterms
                                    ? PInfo[o].prevsolmin
                                    : 0,
                            indices,
                            solmin
                        );
                        pool_boundary_exact[o] = true;
                    }

                    if (*solmin <= 0 || *solmin > ON_minterms) {
                        DBG_ERROR_BLOCK {
                            fprintf(
                                debug_out,
                                "Error: solving the minterm coverage failed.\n"
                            );
                        }
                        free(baseline_indices);
                        free(weights);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }


                    int baseline_solmin = *solmin;
                    memcpy(
                        baseline_indices,
                        indices,
                        (size_t)baseline_solmin * sizeof(int)
                    );
                    bool baseline_exact = pool_boundary_exact[o];

                    const bool probe_due =
                        *solmin > 0 &&
                        adaptive_stopping_supported &&
                        PInfo[o].prevsolmin > 1 &&
                        PInfo[o].prevsolmin <= ON_minterms &&
                        *solmin >= PInfo[o].prevsolmin &&
                        stop_counter[o] + 1 >= STOP_AFTER_EQUALITY;
                    if (probe_due) {
                        const int pre_probe_found = *foundPI;
                        const uint64_t pair_limit =
                            HYBRID_EFFORT_LEVEL >= 2
                                ? UINT64_C(1024)
                                : UINT64_C(128);
                        const int candidate_limit =
                            HYBRID_EFFORT_LEVEL >= 2 ? 128 : 32;
                        PlateauProbeStats probe_stats;

                        if (!plateau_probe_append_candidates(
                            &PInfo[o],
                            ninputs,
                            implicant_words,
                            PInfo[o].previndices,
                            PInfo[o].prevsolmin,
                            bit_index,
                            word_index,
                            shifted_mask,
                            pair_limit,
                            candidate_limit,
                            &probe_stats
                        )) {
                            fprintf(
                                stderr,
                                "Error: plateau probe failed for output %d.\n",
                                o + 1
                            );
                            free(weights);
                            free(baseline_indices);
                            destroy_output_locks(output_locks, noutputs);
                            cleanup(PInfo, buffer);
                            return 1;
                        }

                        if (probe_stats.candidates_appended > 0) {
                            PInfo[o].nofpi[k - 1] = *foundPI;
                            chart = pi_chart_view(&PInfo[o]);
                            free(weights);
                            weights = complete_mmcs_chart
                                ? build_complete_cover_weights(
                                    &PInfo[o],
                                    *foundPI,
                                    ninputs,
                                    implicant_words,
                                    word_index,
                                    shifted_mask,
                                    WEIGHT_PIC
                                )
                                : build_cover_weights(
                                    &PInfo[o],
                                    *foundPI,
                                    k,
                                    WEIGHT_PIC
                                );
                            if (WEIGHT_PIC > 0 && !weights) {
                                fprintf(
                                    stderr,
                                    "Error: Memory allocation failed for "
                                    "plateau-probe cover weights\n"
                                );
                                free(baseline_indices);
                                destroy_output_locks(output_locks, noutputs);
                                cleanup(PInfo, buffer);
                                return 1;
                            }

                            if (SCP_TYPE == 0) {
                                solve_scp_lagrangian(
                                    &chart,
                                    weights,
                                    indices,
                                    solmin,
                                    HYBRID_EFFORT_LEVEL,
                                    baseline_indices,
                                    baseline_solmin,
                                    CERTIFIED_MODE
                                );
                                pool_boundary_exact[o] =
                                    lagrangian_last_run_proved_optimal();
                                print_hybrid_stats(o);
                            }
                            if (SCP_TYPE == 1) {
                                gurobi_multiobjective(
                                    &chart,
                                    weights,
                                    baseline_indices,
                                    baseline_solmin,
                                    indices,
                                    solmin
                                );
                                pool_boundary_exact[o] = true;
                            }

                            if (*solmin >= PInfo[o].prevsolmin) {
                                *foundPI = pre_probe_found;
                                PInfo[o].nofpi[k - 1] = pre_probe_found;
                                chart = pi_chart_view(&PInfo[o]);
                                *solmin = baseline_solmin;
                                memcpy(
                                    indices,
                                    baseline_indices,
                                    (size_t)baseline_solmin * sizeof(int)
                                );
                                pool_boundary_exact[o] = baseline_exact;
                                free(weights);
                                weights = complete_mmcs_chart
                                    ? build_complete_cover_weights(
                                        &PInfo[o],
                                        *foundPI,
                                        ninputs,
                                        implicant_words,
                                        word_index,
                                        shifted_mask,
                                        WEIGHT_PIC
                                    )
                                    : build_cover_weights(
                                        &PInfo[o],
                                        *foundPI,
                                        k,
                                        WEIGHT_PIC
                                    );
                                if (WEIGHT_PIC > 0 && !weights) {
                                    fprintf(
                                        stderr,
                                        "Error: Memory allocation failed "
                                        "while restoring pool weights\n"
                                    );
                                    free(baseline_indices);
                                    destroy_output_locks(
                                        output_locks,
                                        noutputs
                                    );
                                    cleanup(PInfo, buffer);
                                    return 1;
                                }
                            }
                        }

                        DBG_INFO_BLOCK {
                            fprintf(
                                debug_out,
                                "CCUBES_PLATEAU_PROBE level=%d output=%d "
                                "witnesses=%d pairs=%llu compatible=%llu "
                                "generated=%d appended=%d improved=%s pool\n",
                                k,
                                o + 1,
                                probe_stats.private_witnesses,
                                (unsigned long long)
                                    probe_stats.pairs_examined,
                                (unsigned long long)
                                    probe_stats.compatible_pairs,
                                probe_stats.candidates_generated,
                                probe_stats.candidates_appended,
                                *solmin < PInfo[o].prevsolmin
                                    ? "yes"
                                    : "no"
                            );
                        }
                    }

                    /*
                     * Pool discovery is useful only at a boundary which may
                     * terminate the output search. Earlier improvements need
                     * only their ordinary incumbent and avoid the expensive
                     * alternative-preserving solve.
                     */
                    const bool pool_due =
                        *solmin > 0 &&
                        PInfo[o].prevsolmin > 0 &&
                        PInfo[o].prevsolmin <= ON_minterms &&
                        *solmin >= PInfo[o].prevsolmin &&
                        stop_counter[o] + 1 >= STOP_AFTER_EQUALITY;

                    if (pool_due) {
                        baseline_solmin = *solmin;
                        memcpy(
                            baseline_indices,
                            indices,
                            (size_t)baseline_solmin * sizeof(int)
                        );
                        baseline_exact = pool_boundary_exact[o];

                        int pool_limit =
                            automatic_pool_solution_limit(*foundPI);
                        DBG_INFO_BLOCK {
                            fprintf(
                                debug_out,
                                "CCUBES_POOL_BUDGET level=%d output=%d "
                                "chart_cols=%d limit=%d phase=plateau\n",
                                k,
                                o + 1,
                                *foundPI,
                                pool_limit
                            );
                        }

                        if (SCP_TYPE == 0) {
                            solve_scp_lagrangian_pool(
                                &chart,
                                weights,
                                pool_limit,
                                pool_count,
                                pool_solutions,
                                solmin,
                                HYBRID_EFFORT_LEVEL,
                                baseline_indices,
                                baseline_solmin,
                                CERTIFIED_MODE
                            );
                            pool_boundary_exact[o] =
                                baseline_exact ||
                                lagrangian_last_run_proved_optimal();
                            print_hybrid_stats(o);
                        }

                        if (SCP_TYPE == 1) {
                            gurobi_solution_pool(
                                &chart,
                                pool_limit,
                                weights,
                                baseline_indices,
                                baseline_solmin,
                                pool_count,
                                pool_solutions,
                                solmin
                            );
                            pool_boundary_exact[o] = true;
                        }

                        if (
                            *solmin <= 0 ||
                            *solmin > baseline_solmin ||
                            (*pool_count <= 0 && *solmin < baseline_solmin)
                        ) {
                            clear_output_solution_pool(&PInfo[o]);
                            *solmin = baseline_solmin;
                            pool_boundary_exact[o] = baseline_exact;
                        }

                        if (*solmin == baseline_solmin) {
                            if (!pool_prepend_solution(
                                pool_solutions,
                                pool_count,
                                pool_limit,
                                baseline_indices,
                                baseline_solmin
                            )) {
                                fprintf(
                                    stderr,
                                    "Error: unable to retain the ordinary "
                                    "incumbent in the solution pool\n"
                                );
                                free(baseline_indices);
                                free(weights);
                                destroy_output_locks(
                                    output_locks,
                                    noutputs
                                );
                                cleanup(PInfo, buffer);
                                return 1;
                            }
                        }
                    } else {
                        DBG_INFO_BLOCK {
                            fprintf(
                                debug_out,
                                "CCUBES_POOL_DEFER level=%d output=%d "
                                "reason=primary-improved\n",
                                k,
                                o + 1
                            );
                        }
                    }

                    clock_gettime(CLOCK_MONOTONIC, &endg);
                    execution_time =
                        (endg.tv_sec - startg.tv_sec) +
                        (endg.tv_nsec - startg.tv_nsec) / 1e9;

                    k_scp_time += execution_time;

                    if (complete_mmcs_chart) {
                        certification_chart_solve_time += execution_time;
                    }

                    pool_execution_time[o] = execution_time;

                    free(baseline_indices);
                    free(weights);
                }

                for (int i = 0; i < ON_minterms; i++) {
                    PInfo[o].last_index[i] = PInfo[o].k_last_index[i];
                }

            } // end of outputs loop solving SCP

            /* Coordinate the per-output pools against their actual joint union. */
            int *chosen_idx = (int*)calloc((size_t)noutputs, sizeof(int));
            if (!chosen_idx) {
                fprintf(stderr, "Error: Memory allocation failed for chosen_idx\n");
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }
            PoolSelectionStats pool_stats;
            if (
                !select_joint_pool_solutions(
                    PInfo,
                    noutputs,
                    implicant_words,
                    chosen_idx,
                    &pool_stats
                )
            ) {
                fprintf(stderr, "Error: coordinated pool selection failed\n");
                free(chosen_idx);
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }

            // Copy chosen solutions into indices for each output
            for (int o = 0; o < noutputs; ++o) {
                int *prevsolmin = &PInfo[o].prevsolmin;
                int *solmin      = &PInfo[o].solmin;
                int *previndices =  PInfo[o].previndices;
                int *indices     =  PInfo[o].indices;

                int pool_count_val = PInfo[o].pool_count;
                int solmin_o = PInfo[o].solmin;

                /* Only update stop logic if a valid solution exists at this k */
                if (solmin_o > 0) {
                    const bool output_was_searching = !PInfo[o].stop_search;
                    bool has_choice = !(pool_count_val <= 0 || chosen_idx[o] < 0);
                    if (has_choice) {
                        int *src = PInfo[o].pool_solutions[chosen_idx[o]];
                        for (int i = 0; i < solmin_o; ++i) {
                            indices[i] = src[i];
                        }
                    } else if (*prevsolmin == solmin_o) {
                        for (int i = 0; i < solmin_o; ++i) {
                            indices[i] = previndices[i];
                        }
                    }

                    bool plateau_triggered = false;
                    bool inspect_equal_pool = false;
                    if (*solmin < *prevsolmin) {
                        *prevsolmin = *solmin;
                        for (int i = 0; i < *solmin; i++) {
                            previndices[i] = indices[i];
                        }

                        stop_counter[o] = 0;

                    } else {
                        if (*solmin == *prevsolmin) {
                            /* Preserve the incumbent until tied covers are diagnosed. */
                            inspect_equal_pool = true;
                        } else {
                            /* A worse boundary incumbent cannot replace the previous one. */
                            for (int i = 0; i < *prevsolmin; i++) {
                                indices[i] = previndices[i];
                            }
                            *solmin = *prevsolmin;

                            /*
                            The current pool contains covers of the discarded,
                            larger cardinality.  It must not survive into a
                            later joint-pool pass, where solmin now describes
                            the shorter retained incumbent.
                            */
                            clear_output_solution_pool(&PInfo[o]);
                        }
                        stop_counter[o]++;

                        plateau_triggered = stop_counter[o] >= STOP_AFTER_EQUALITY;
                    }

                    if (adaptive_stopping_supported && !certified_blocking_observe_plateau(
                        &blocking_stop_states[o],
                        REPORT_BLOCKING_DIAGNOSTIC,
                        adaptive_warning_enabled,
                        plateau_triggered,
                        stderr,
                        o + 1,
                        &PInfo[o],
                        ninputs,
                        k,
                        indices,
                        *solmin,
                        pool_boundary_exact[o],
                        inspect_equal_pool
                    )) {
                        fprintf(stderr, "Error: blocking diagnostic failed for output %d.\n", o + 1);
                        free(chosen_idx);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    if (inspect_equal_pool) {
                        /* Retain the final coordinated/diagnostic-safe tied cover. */
                        for (int i = 0; i < *solmin; ++i) {
                            previndices[i] = indices[i];
                        }
                    }

                    PInfo[o].stop_search = certified_stop_policy_decision(
                        certified_states ? &certified_states[o] : NULL,
                        CERTIFIED_MODE,
                        plateau_triggered,
                        k,
                        *solmin,
                        pool_boundary_exact[o],
                        stderr,
                        o + 1
                    );

                    if (complete_mmcs_chart) {
                        fprintf(
                            stderr,
                            "CCUBES_CERTIFICATION_CHART output=%d rows=%d "
                            "primes=%d cover=%d boundary_exact=%d "
                            "seconds=%.6f\n",
                            o + 1,
                            PInfo[o].ON_minterms,
                            PInfo[o].foundPI,
                            *solmin,
                            pool_boundary_exact[o] ? 1 : 0,
                            pool_execution_time[o]
                        );
                    }

                    DBG_INFO_BLOCK {
                        if (*solmin > 0 && output_was_searching) {
                            fprintf(
                                debug_out,
                                "[pool] Output %d (SCP %.3fs) solution (%d): ",
                                o + 1,
                                pool_execution_time[o],
                                *solmin
                            );
                            for (int i = 0; i < *solmin; ++i) {
                                fprintf(debug_out, "%d ", indices[i] + 1);
                            }
                            fprintf(
                                debug_out,
                                "%s\n",
                                PInfo[o].stop_search
                                    ? "-- stopping search"
                                    : plateau_triggered
                                        ? "-- plateau; continuing certification"
                                        : ""
                            );
                        }
                    }
                }
            }

            if (!measure_selected_pool_solutions(
                PInfo,
                noutputs,
                implicant_words,
                &pool_stats
            )) {
                fprintf(stderr, "Error: coordinated pool measurement failed\n");
                free(chosen_idx);
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }

            DBG_INFO_BLOCK {
                pool_stats.retained_shared_cubes = count_retained_shared_cubes(
                    PInfo,
                    noutputs,
                    implicant_words
                );
                fprintf(
                    debug_out,
                    "CCUBES_POOL level=%d active_outputs=%d generated=%d "
                    "valuable=%d discarded=%d candidates=%d "
                    "retained_shared=%d pool_shared=%d connections=%d "
                    "selected_rows=%d input_literals=%d "
                    "selected_shared=%d savings=%d selection=%s\n",
                    k,
                    pool_stats.active_outputs,
                    pool_stats.generated_pool_solutions,
                    pool_stats.valuable_pool_solutions,
                    pool_stats.discarded_pool_solutions,
                    pool_stats.total_pool_solutions,
                    pool_stats.retained_shared_cubes,
                    pool_stats.pool_shared_cubes,
                    pool_stats.output_connections,
                    pool_stats.selected_distinct_cubes,
                    pool_stats.selected_input_literals,
                    pool_stats.selected_shared_cubes,
                    pool_stats.sharing_savings,
                    pool_stats.selection_exact ? "exact" : "local"
                );
            }

            free(chosen_idx);

        } // end of searching for solutions with pooling

        clock_gettime(CLOCK_MONOTONIC, &endk);
        execution_time =
            (endk.tv_sec - startk.tv_sec) +
            (endk.tv_nsec - startk.tv_nsec) / 1e9;

        DBG_INFO_BLOCK {
            fprintf(debug_out, "k-level execution completed in %.3f (%.3f) seconds\n", execution_time, k_scp_time);
            fprintf(
                debug_out,
                "CCUBES_TIMING level=%d pi_generation=%.6f scp=%.6f other=%.6f total=%.6f\n",
                k,
                k_pi_generation_time,
                k_scp_time,
                execution_time - k_pi_generation_time - k_scp_time,
                execution_time
            );
        }

        scp_time += k_scp_time;

        bool stop = true;
        for (int o = 0; o < noutputs; o++) {
            stop &= PInfo[o].stop_search;
        }
        if (stop) break;
        if (complete_mmcs_chart) break;

    } // end of k loop

    DBG_TRACE_BLOCK {
        fprintf(debug_out, "max shared: %d\n", max_shared);
        fprintf(debug_out, "\n--- END ---\n");
    }

    for (int o = 0; o < noutputs; ++o) {
        if (CERTIFIED_MODE && !PInfo[o].stop_search) {
            fprintf(
                stderr,
                "Error: no global optimality certificate was established "
                "for output %d.\n",
                o + 1
            );
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            debug_close();
            return 1;
        }
    }

    if (POOL_MAX > 1) {
        int *chosen_idx = (int *)calloc((size_t)noutputs, sizeof(int));
        PoolSelectionStats final_pool_stats;
        if (
            !chosen_idx ||
            !select_final_joint_pool_solutions(
                PInfo,
                noutputs,
                implicant_words,
                chosen_idx,
                &final_pool_stats
            )
        ) {
            fprintf(stderr, "Error: final coordinated pool selection failed\n");
            free(chosen_idx);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            debug_close();
            return 1;
        }

        for (int o = 0; o < noutputs; ++o) {
            if (chosen_idx[o] < 0) continue;
            int *selected = PInfo[o].pool_solutions[chosen_idx[o]];
            for (int i = 0; i < PInfo[o].solmin; ++i) {
                PInfo[o].indices[i] = selected[i];
                PInfo[o].previndices[i] = selected[i];
            }
        }

        if (!measure_selected_pool_solutions(
            PInfo,
            noutputs,
            implicant_words,
            &final_pool_stats
        )) {
            fprintf(stderr, "Error: final coordinated pool measurement failed\n");
            free(chosen_idx);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            debug_close();
            return 1;
        }

        DBG_INFO_BLOCK {
            fprintf(
                debug_out,
                "CCUBES_POOL_FINAL generated=%d valuable=%d discarded=%d "
                "connections=%d selected_rows=%d input_literals=%d "
                "selected_shared=%d savings=%d selection=%s\n",
                final_pool_stats.generated_pool_solutions,
                final_pool_stats.valuable_pool_solutions,
                final_pool_stats.discarded_pool_solutions,
                final_pool_stats.output_connections,
                final_pool_stats.selected_distinct_cubes,
                final_pool_stats.selected_input_literals,
                final_pool_stats.selected_shared_cubes,
                final_pool_stats.sharing_savings,
                final_pool_stats.selection_exact ? "exact" : "local"
            );
        }

        free(chosen_idx);
    }

    free(certified_states);

    for (int o = 0; o < noutputs; o++) {
        int solmin = PInfo[o].solmin > 0 ? PInfo[o].solmin : 1; // at least one solution term is needed
        PInfo[o].solution = (int *) calloc(solmin * ninputs, sizeof(int));

        for (int c = 0; c < PInfo[o].solmin; c++) {
            for (int r = 0; r < ninputs; r++) {
                int value = 0;
                int position = r * PInfo[o].solmin + c;

                if (PInfo[o].implicants_pos[PInfo[o].indices[c] * implicant_words + word_index[r]] & shifted_mask[r]) {
                    value = 1 + (int)((PInfo[o].implicants_val[PInfo[o].indices[c] * implicant_words + word_index[r]] >> bit_index[r]) & VALUE_BIT_MASK);
                    PInfo[o].solution[position] = value; // transposed
                }
            }
        }
    }

    if (DST_FILE == NULL) {
        if (RESUME_PATH) {
            // Derive sensible default from checkpoint name, always ending with .pla
            const char *base = strrchr(RESUME_PATH, '/');
            base = base ? base + 1 : RESUME_PATH;

            const char *chk_prefix = "chk_";
            size_t chk_len = strlen(chk_prefix);

            // Determine the meaningful part after removing chk_ if present
            const char *rest = (strncmp(base, chk_prefix, chk_len) == 0) ? (base + chk_len) : base;

            // Strip extension from rest
            const char *dot = strrchr(rest, '.');
            size_t stem_len = dot ? (size_t)(dot - rest) : strlen(rest);

            const char *out_prefix = "ccubes_";
            size_t out_len = strlen(out_prefix) + stem_len + 4 /*.pla*/ + 1;
            char *outname = (char *)malloc(out_len);
            if (outname) {
                strcpy(outname, out_prefix);
                strncat(outname, rest, stem_len);
                outname[strlen(out_prefix) + stem_len] = '\0';
                strcat(outname, ".pla");
                DST_FILE = outname; // freed on process exit
            }
        } else if (SRC_FILE) {
            const char *prefix = "ccubes_";
            char *filename = prefix_basename(SRC_FILE, prefix);
            if (filename) {
                DST_FILE = filename;
                // Note: filename memory will be cleaned up along with other resources
            }
        }
    }

    if (DST_FILE == NULL) {
        fprintf(stderr, "Error: failed to determine destination .pla filename.\n");
        destroy_output_locks(output_locks, noutputs);
        cleanup(PInfo, buffer);
        debug_close();
        return 1;
    }

    if (!run_log_check_path(DST_FILE)) {
        destroy_output_locks(output_locks, noutputs);
        cleanup(PInfo, buffer);
        debug_close();
        return 1;
    }
    write_pla_file(DST_FILE, PInfo);

    // Free allocated memory
    destroy_output_locks(output_locks, noutputs);
    cleanup(PInfo, buffer);
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate and log execution time (accumulated across resumes)
    execution_time =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    double total_exec_time = BASE_ELAPSED + execution_time;
    double total_scp_time  = BASE_SCP + scp_time;

    fprintf(stderr, "Execution completed in %.3f (%.3f SCP) seconds\n", total_exec_time, total_scp_time);
    fprintf(
        stderr,
        "CCUBES_TIMING_TOTAL pi_generation=%.6f scp=%.6f other=%.6f total=%.6f\n",
        pi_generation_time,
        total_scp_time,
        total_exec_time - pi_generation_time - total_scp_time,
        total_exec_time
    );
    if (CERTIFIED_MODE) {
        double certification_other_time =
            total_exec_time -
            certification_enumeration_wall_time -
            certification_chart_solve_time;
        if (certification_other_time < 0.0) {
            certification_other_time = 0.0;
        }
        fprintf(
            stderr,
            "CCUBES_CERTIFICATION_TIMING phase=summary "
            "enumeration_wall_seconds=%.6f "
            "enumeration_worker_seconds=%.6f "
            "chart_solve_seconds=%.6f other_seconds=%.6f "
            "total_seconds=%.6f outputs=%d primes=%llu\n",
            certification_enumeration_wall_time,
            certification_enumeration_worker_time,
            certification_chart_solve_time,
            certification_other_time,
            total_exec_time,
            noutputs,
            (unsigned long long)certification_prime_count
        );
    }

    lock_stats_set_generation_seconds(pi_generation_time);
    lock_stats_report(stderr);
    lock_stats_free();

    DBG_INFO_BLOCK {
        fprintf(debug_out, "all good.\n");
    }

    debug_close();
    return 0;

}
