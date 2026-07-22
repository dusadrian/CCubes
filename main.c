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
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <stdatomic.h>
#include "main.h"
#include "certified_stop.h"
#include "checkpoint.h"
#include "ccubes_threads.h"

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
    int *max_shared;
    int increase;
    int *multiplier;
    bool deterministic_order;
    double time_limit_sec;
    double base_elapsed;
    struct timespec start_time;
    atomic_bool *time_up;
    double *time_up_elapsed;
    atomic_uint_fast64_t *last_task_reached;
    ccubes_mutex *state_lock;
} PIWorkerContext;

static void destroy_output_locks(ccubes_mutex *locks, int noutputs) {
    if (!locks) return;
    for (int o = 0; o < noutputs; o++) {
        ccubes_mutex_destroy(&locks[o]);
    }
    free(locks);
}

static bool env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value || !*value) return false;

    return (
        strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 &&
        strcmp(value, "FALSE") != 0 &&
        strcmp(value, "no") != 0 &&
        strcmp(value, "NO") != 0
    );
}

static bool parse_int_strict(const char *text, int *value) {
    if (!text || !*text || !value) return false;

    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    if (parsed < INT_MIN || parsed > INT_MAX) return false;

    *value = (int)parsed;
    return true;
}

static bool parse_nonnegative_double(const char *text, double *value) {
    if (!text || !*text || !value) return false;

    char *end = NULL;
    errno = 0;
    double parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') return false;
    if (!isfinite(parsed) || parsed < 0.0) return false;

    *value = parsed;
    return true;
}

static bool parse_hybrid_effort_level(const char *text, int *level) {
    int parsed = 0;
    if (!parse_int_strict(text, &parsed)) return false;
    if (parsed < 0 || parsed > 2) return false;
    *level = parsed;
    return true;
}

static void debug_print_lagrangian_stats(int output_index) {
    DBG_INFO_BLOCK {
        const LagrangianStats *stats = lagrangian_last_stats();
        if (!stats || stats->stop_reason == LAGR_STOP_NOT_RUN) return;

        fprintf(
            debug_out,
            "Lagrangian output %d: rows=%d cols=%d UB=%d ",
            output_index + 1,
            stats->rows,
            stats->cols,
            stats->best_ub
        );

        if (stats->best_lb == INT_MIN) {
            fprintf(debug_out, "LB=- gap=- ");
        } else {
            fprintf(
                debug_out,
                "LB=%d gap=%d ",
                stats->best_lb,
                stats->gap
            );
        }

        fprintf(
            debug_out,
            "bestZLB=%.6f lastZLB=%.6f iterations=%d stop=%s%s\n",
            stats->best_zlb,
            stats->last_zlb,
            stats->iterations,
            lagrangian_stop_reason_name(stats->stop_reason),
            stats->pool_mode ? " pool" : ""
        );
    }
}

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

    for (uint64_t task = start; task < end; task += stride) {
        if (atomic_load_explicit(ctx->time_up, memory_order_acquire)) {
            break;
        }

        if (ctx->time_limit_sec > 0.0 && (task & 0x3FFull) == 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed =
                (now.tv_sec - ctx->start_time.tv_sec) +
                (now.tv_nsec - ctx->start_time.tv_nsec) / 1e9;

            if (elapsed >= ctx->time_limit_sec) {
                bool expected = false;
                if (
                    atomic_compare_exchange_strong_explicit(
                        ctx->time_up,
                        &expected,
                        true,
                        memory_order_acq_rel,
                        memory_order_acquire
                    )
                ) {
                    ccubes_mutex_lock(ctx->state_lock);
                    *ctx->time_up_elapsed = ctx->base_elapsed + elapsed;
                    atomic_store_explicit(ctx->last_task_reached, task, memory_order_release);
                    ccubes_mutex_unlock(ctx->state_lock);
                }
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
            ctx->max_shared,
            ctx->increase,
            ctx->multiplier,
            ctx->deterministic_order
        );

        if (error) {
            fprintf(
                stderr,
                "Error: process_task failed for task %llu at k=%d\n",
                (unsigned long long)task,
                ctx->k
            );
            atomic_store_explicit(ctx->time_up, true, memory_order_release);
            break;
        }
    }
}

void help() {
    printf("Usage: ccubes [options] source.pla [dest.pla]\n");
    printf("Options:\n");
    printf("  -b<number>           : bits per word, either 8, 16, 32, 64 (default) or 128\n");
    printf("  -t<number>           : number of CPU worker threads to use\n");
    printf("  -w<number>           : weights applied to the prime implicants:\n");
    printf("                           0 no weight\n");
    printf("                           1 (default) weight based on complexity levels k\n");
    printf("                           2 additional weight if shared between outputs\n");
    printf("  -s<number>           : how to solve the covering problem:\n");
    printf("                           0 (default) bundled hybrid solver\n");
    printf("                             (presolve + Lagrangian bounds + bounded exact search)\n");
    printf("                           1 Gurobi exact\n");
    printf("  -e<number>           : hybrid solver effort level:\n");
    printf("                           0 (default) fastest, bounded strong finish\n");
    printf("                           1 stronger bounds, more time\n");
    printf("                           2 best bound mode, adaptive bundle portfolio with bounded strong finish\n");
    printf("  -d                   : deterministic PI ordering\n");
    printf("  -g                   : print the adaptive blocking diagnostic at the first plateau\n");
    printf("  -c                   : require certified exact stopping\n");
    printf("                         (default: adaptive warning, then certify only when warned)\n");
    printf("  -p<number>           : decide from a pool of up to <number> equally optimal solutions\n");
    printf("  -l<sec>[=<file>]     : time limit to save a checkpoint in the <file>\n");
    printf("  -r=<file>            : resume from checkpoint file\n");
    printf("  -i<level>=<file>     : inspect checkpoint (print progress and metadata)\n");
    printf("                           0 (default) progress report\n");
    printf("                           1 complete metadata about each output\n");
    printf("  -dbg<level>[=<file>] : incremental debug information\n");
    printf("                           0 (default) errors + warnings\n");
    printf("                           1 errors + warnings + info\n");
    printf("                           2 everything (trace)\n");
    printf("  -h, --help           : show this help message\n");
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        help();
        return 1;
    }

    bool gurobi_ok = false;

    // Record start time for execution timing
    struct timespec start, end, startk, endk, startg, endg;
    double execution_time;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // defaults
    int search_level = 1;
    const int STOP_AFTER_EQUALITY = 1;
    int BITS_PER_WORD = 64;
    int THREADS = 0; // max by default
    bool THREADS_FORCED = false; // set to true if -t is provided
    int WEIGHT_PIC = 1;
    int SCP_TYPE = 0;
    int HYBRID_EFFORT_LEVEL = 0;
    int POOL_MAX = 1; // collect up to this many solutions
    bool DETERMINISTIC_PI_ORDER = env_flag_enabled("CCUBES_DETERMINISTIC");
    bool CERTIFIED_MODE = false;
    bool REPORT_BLOCKING_DIAGNOSTIC = false;
    char *SRC_FILE = NULL;
    char *DST_FILE = NULL;
    // resume timing bases
    double BASE_ELAPSED = 0.0;
    double BASE_SCP = 0.0;

    // checkpoint/time limit
    double TIME_LIMIT_SEC = 0.0;
    char *CHK_SAVE_PATH = NULL;
    char *RESUME_PATH = NULL;
    char *INFO_PATH = NULL; // -i: inspect checkpoint file
    int RESUME_K = -1;
    // resume progress for current k
    uint64_t RESUME_LAST_TASK = 0ull;
    bool HAS_RESUME_LAST_TASK = false;

    int info_level = 0;

    // parse arguments
    for (int i = 1; i < argc; i++) {
        if (
            strncmp(argv[i], "-dbg", 4) == 0 ||
            strncmp(argv[i], "-debug", 6) == 0
        ) {
            size_t prefix_length = strncmp(argv[i], "-dbg", 4) == 0 ? 4u : 6u;
            char *opt = argv[i] + prefix_length;
            char *eq = strchr(opt, '=');
            int debug_level = 0;
            const char *file = NULL;

            if (eq) {
                *eq = '\0';
                debug_level = atoi(opt);
                file = eq + 1;
            } else {
                debug_level = atoi(opt);
            }

            if (debug_level < 0 || debug_level > 2) {
                fprintf(stderr, "Invalid debug level: %d (must be 0-2)\n", debug_level);
                help();
                return 1;
            }
            debug_init(file, debug_level);
        } else if (strcmp(argv[i], "-d") == 0) {
            DETERMINISTIC_PI_ORDER = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            CERTIFIED_MODE = true;
        } else if (strcmp(argv[i], "-g") == 0) {
            REPORT_BLOCKING_DIAGNOSTIC = true;
        } else if (strncmp(argv[i], "-b", 2) == 0) {
            BITS_PER_WORD = atoi(argv[i] + 2);
            if (
                BITS_PER_WORD != 8 && BITS_PER_WORD != 16 &&
                BITS_PER_WORD != 32 && BITS_PER_WORD != 64 &&
                BITS_PER_WORD != 128
            ) {
                fprintf(stderr, "Invalid bits per word: %d (must be 8, 16, 32, 64, or 128)\n", BITS_PER_WORD);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-w", 2) == 0) {
            WEIGHT_PIC = atoi(argv[i] + 2);
            if (WEIGHT_PIC < 0 || WEIGHT_PIC > 2) {
                fprintf(stderr, "Invalid weight option: %d (must be 0, 1, or 2)\n", WEIGHT_PIC);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-s", 2) == 0) {
            SCP_TYPE = atoi(argv[i] + 2);
            if (SCP_TYPE != 0 && SCP_TYPE != 1) {
                fprintf(stderr, "Invalid SCP solver: %d (must be 0 or 1)\n", SCP_TYPE);
                help();
                return 1;
            }
        } else if (strcmp(argv[i], "-e") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -e requires an effort level (0, 1, or 2)\n");
                return 1;
            }
            if (!parse_hybrid_effort_level(argv[++i], &HYBRID_EFFORT_LEVEL)) {
                fprintf(stderr, "Invalid hybrid effort level: %s (must be 0, 1, or 2)\n", argv[i]);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-e", 2) == 0) {
            if (!parse_hybrid_effort_level(argv[i] + 2, &HYBRID_EFFORT_LEVEL)) {
                fprintf(stderr, "Invalid hybrid effort level: %s (must be 0, 1, or 2)\n", argv[i] + 2);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-t", 2) == 0) {
            if (!parse_int_strict(argv[i] + 2, &THREADS) || THREADS < 0) {
                fprintf(stderr, "Invalid thread count: %s (must be a nonnegative integer)\n", argv[i] + 2);
                help();
                return 1;
            }
            THREADS_FORCED = true;
        } else if (strncmp(argv[i], "-p", 2) == 0) {
            POOL_MAX = atoi(argv[i] + 2);
            if (POOL_MAX < 1) POOL_MAX = 1;
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            char *opt = argv[i] + 2;  // string after "-l"
            char *eq  = strchr(opt, '=');
            if (eq) {
                *eq = '\0';
                CHK_SAVE_PATH  = eq + 1;
            }
            if (!parse_nonnegative_double(opt, &TIME_LIMIT_SEC)) {
                fprintf(stderr, "Invalid time limit: %s (must be nonnegative seconds)\n", opt);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-r", 2) == 0) {
            // Support -r=<file> (preferred) and legacy -r<file>
            if (argv[i][2] == '=') {
                RESUME_PATH = argv[i] + 3;
            } else if (argv[i][2] != '\0') {
                RESUME_PATH = argv[i] + 2;
            } else if (i + 1 < argc) {
                RESUME_PATH = argv[++i];
            } else {
                fprintf(stderr, "Error: -r requires a file path (use -r=<file>)\n");
                return 1;
            }
        } else if (strncmp(argv[i], "-i", 2) == 0) {
            char *opt = argv[i] + 2;  // string after "-i"
            char *eq  = strchr(opt, '=');
            if (eq) {
                *eq = '\0';
                info_level = atoi(opt);
                INFO_PATH  = eq + 1;
            } else {
                fprintf(stderr, "Error: -i requires a file path (use -i<level>=<file>)\n");
                return 1;
            }
            if (info_level < 0 || info_level > 1) {
                fprintf(stderr, "Invalid info level: %d (must be 0 or 1)\n", info_level);
                help();
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help();
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            help();
            return 1;
        } else if (!SRC_FILE) {
            SRC_FILE = argv[i];
        } else if (!DST_FILE) {
            DST_FILE = argv[i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            help();
            return 1;
        }
    }

    #ifdef HAVE_GUROBI
        if (SCP_TYPE == 1) {
            gurobi_ok = gurobi_license_is_valid();
        }
    #endif


    if (SCP_TYPE == 1) {
        if (!gurobi_ok) {
            SCP_TYPE = 0;
            printf("Gurobi not available, falling back to the bundled hybrid solver.\n");
        }
    }

    if (POOL_MAX > 1) {
        WEIGHT_PIC = 2;
    }

    // Inspect checkpoint and exit
    if (INFO_PATH) {
        print_info(INFO_PATH, info_level);
        return 0;
    }

    if (!SRC_FILE && !RESUME_PATH) {
        fprintf(stderr, "Error: source .pla file is required.\n");
        help();
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
    int *chk_coverage_horizon = NULL;
    bool chk_certified_mode = false;
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
        uint64_t chk_last_task = 0ull;

        if (
            load_checkpoint(
                RESUME_PATH,
                &PInfo,
                &ninputs,
                &noutputs,
                &chk_bits,
                &chk_value_bit_width,
                &chk_implicant_words,
                &RESUME_K,
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
                &chk_last_task,
                &chk_certified_mode,
                &chk_coverage_horizon
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

        (void)chk_MAX_LEVELS;
        CERTIFIED_MODE = CERTIFIED_MODE || chk_certified_mode;
        WEIGHT_PIC = chk_WEIGHT_PIC;
        SCP_TYPE = chk_SCP_TYPE;
        POOL_MAX = chk_POOL_MAX;
        chk_value_bit_width_saved = chk_value_bit_width;
        chk_implicant_words_saved = chk_implicant_words;

        // Resume timing bases
        BASE_ELAPSED = chk_elapsed_total;
        BASE_SCP = chk_elapsed_scp;

        // Resume the interrupted generation level recorded in the checkpoint.
        search_level = RESUME_K;
        // carry forward last_task information for this k
        RESUME_LAST_TASK = chk_last_task;
        HAS_RESUME_LAST_TASK = true;

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
            printf("Error: Invalid .pla file or no inputs/outputs found.\n");
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

    if (!certified_model_supported(PInfo, ninputs, noutputs)) {
        fprintf(
            stderr,
            "Error: adaptive and certified stopping require binary, fully specified input rows "
            "with nonempty ON and OFF sets for every output.\n"
        );
        free(chk_stop_counter);
        free(chk_coverage_horizon);
        ThreadBuffer *empty_buffer[1] = {NULL};
        cleanup(PInfo, empty_buffer);
        return 1;
    }

    certified_states = calloc((size_t)noutputs, sizeof(CertifiedStopState));
    if (!certified_states) {
        fprintf(stderr, "Error: stopping-state allocation failed.\n");
        free(chk_stop_counter);
        free(chk_coverage_horizon);
        ThreadBuffer *empty_buffer[1] = {NULL};
        cleanup(PInfo, empty_buffer);
        return 1;
    }

    for (int o = 0; o < noutputs; ++o) {
        certified_stop_state_reset(&certified_states[o]);
        if (chk_coverage_horizon) {
            certified_states[o].coverage_horizon = chk_coverage_horizon[o];
        }
        if (CERTIFIED_MODE && !certified_stop_state_prepare(
            &certified_states[o],
            &PInfo[o],
            ninputs
        )) {
            fprintf(stderr, "Error: certified-horizon initialization failed.\n");
            free(chk_stop_counter);
            free(chk_coverage_horizon);
            free(certified_states);
            ThreadBuffer *empty_buffer[1] = {NULL};
            cleanup(PInfo, empty_buffer);
            return 1;
        }
        if (CERTIFIED_MODE) {
            DBG_INFO_BLOCK {
                fprintf(
                    debug_out,
                    "Certified output %d: agreement horizon Amax=%d, cover LB=%d\n",
                    o + 1,
                    certified_states[o].agreement_horizon,
                    certified_states[o].cover_lower_bound
                );
            }
        }
    }
    free(chk_coverage_horizon);
    chk_coverage_horizon = NULL;

    uint64_t VALUE_BIT_MASK = (1ULL << value_bit_width) - 1ULL;

    // Ensure pichart_words are aligned with BITS_PER_WORD (already set on load for resume)
    PInfo[0].pichart_words = (PInfo[0].ON_minterms + BITS_PER_WORD - 1) / BITS_PER_WORD;
    for (int o = 0; o < noutputs; o++) {
        PInfo[o].pichart_words = (PInfo[o].ON_minterms + BITS_PER_WORD - 1) / BITS_PER_WORD;
    }



    int increase = 100000; // how much to increase the size of the arrays when needed

    // many PIs will have the same coverage, but they don't necessarily cover the same minterms
    // to employ row dominance when solving the coverage matrix, we need to compare the coverage of
    // the current PI with the coverage of the previous PIs. If this PI survives the comparison, its
    // coverage has to be added in the "covered" vector, at the last index of the PI coverage with
    // the same number of (covered) minterms
    // int last_index[ON_minterms]; // descending order

    int layer_weights[ninputs + 1];
    layer_weights[0] = 0; // k basically starts with 1, so its k-layer weight will start on position 1


    int available_threads = ccubes_default_thread_count();
    if (THREADS_FORCED) {
        if (THREADS < 1) THREADS = 1;
        if (THREADS > available_threads) THREADS = available_threads;
    } else {
        THREADS = available_threads;
    }
    THREADS = ccubes_effective_thread_count(THREADS);
    ccubes_set_thread_count(THREADS);

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

        // preallocating for a moderate number; grows dynamically as needed
        if (!RESUME_PATH) {
            PInfo[o].estimPI = 500000;
        } else {
            if (PInfo[o].estimPI < PInfo[o].foundPI) PInfo[o].estimPI = PInfo[o].foundPI; // minimal capacity
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

        if (!PInfo[o].last_index) PInfo[o].last_index = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].k_last_index) PInfo[o].k_last_index = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].last_index || !PInfo[o].k_last_index) {
            fprintf(stderr, "Error: Memory allocation failed for index arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!PInfo[o].pichart) {
            PInfo[o].pichart = (int *) calloc(PInfo[o].estimPI * ON_minterms, sizeof(int));
            if (!PInfo[o].pichart) {
                fprintf(stderr, "Error: Memory allocation failed for pichart\n");
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        PInfo[o].pichart_words = (ON_minterms + BITS_PER_WORD - 1) / BITS_PER_WORD; // Words needed per coverage matrix columns
        if (!PInfo[o].pichart_pos) {
            PInfo[o].pichart_pos = (uint64_t *) calloc(PInfo[o].estimPI * PInfo[o].pichart_words, sizeof(uint64_t));
            if (!PInfo[o].pichart_pos) {
                fprintf(stderr, "Error: Memory allocation failed for pichart_pos\n");
                cleanup(PInfo, buffer);
                return 1;
            }
        }

        if (!PInfo[o].implicants_pos) PInfo[o].implicants_pos = (uint64_t *) calloc(PInfo[o].estimPI * implicant_words, sizeof(uint64_t));
        if (!PInfo[o].implicants_val) PInfo[o].implicants_val = (uint64_t *) calloc(PInfo[o].estimPI * implicant_words, sizeof(uint64_t));
        if (!PInfo[o].implicants_pos || !PInfo[o].implicants_val) {
            fprintf(stderr, "Error: Memory allocation failed for implicants arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!PInfo[o].shared) PInfo[o].shared = (int *) calloc(PInfo[o].estimPI, sizeof(int));
        if (!PInfo[o].covsum) PInfo[o].covsum = (int *) calloc(PInfo[o].estimPI, sizeof(int));
        if (!PInfo[o].shared || !PInfo[o].covsum) {
            fprintf(stderr, "Error: Memory allocation failed for shared/covsum arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!RESUME_PATH) {
            PInfo[o].foundPI = 0;
            PInfo[o].solmin = 0;
        }

        // the previous (level k - 1), minimum number of PIs to solve the coverage matrix
        if (!RESUME_PATH) {
            PInfo[o].prevsolmin = ON_minterms + 1; // Initialization
        }

        // the positions of the PIs solving the coverage matrix
        // this vector can never be lengthier than the number of ON minterms (ON_minterms)
        if (!PInfo[o].previndices) PInfo[o].previndices = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].indices)     PInfo[o].indices     = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].previndices || !PInfo[o].indices) {
            fprintf(stderr, "Error: Memory allocation failed for indices arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        if (!RESUME_PATH) {
            PInfo[o].ON_set_covered = false;
        }

        if (!PInfo[o].cov_word_index)   PInfo[o].cov_word_index   = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].shifted_cov_mask) PInfo[o].shifted_cov_mask = (uint64_t *) calloc(ON_minterms, sizeof(uint64_t));
        if (!PInfo[o].cov_word_index || !PInfo[o].shifted_cov_mask) {
            fprintf(stderr, "Error: Memory allocation failed for coverage arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        for (int r = 0; r < ON_minterms; r++) {
            PInfo[o].cov_word_index[r] = r / BITS_PER_WORD;
            PInfo[o].shifted_cov_mask[r] = (1ULL << (r % BITS_PER_WORD));
        }

        // If resuming, restore stop counter from checkpoint
        if (chk_stop_counter) {
            stop_counter[o] = chk_stop_counter[o];
        } else {
            stop_counter[o] = 0;
        }
        if (RESUME_PATH && stop_counter[o] > 0) {
            blocking_stop_states[o].reported = true;
            if (!CERTIFIED_MODE && !PInfo[o].stop_search) {
                blocking_stop_states[o].certification_required = true;
                if (!certified_stop_state_prepare(
                    &certified_states[o],
                    &PInfo[o],
                    ninputs
                )) {
                    fprintf(stderr, "Error: failed to restore adaptive certification state.\n");
                    free(certified_states);
                    cleanup(PInfo, buffer);
                    return 1;
                }
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

        // Preserve stop status when resuming; otherwise initialize it.
        if (!RESUME_PATH) {
            PInfo[o].stop_search = ON_minterms == 0; // if there are no ON-set minterms, then stop searching for PIs
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
            int ON_minterms = PInfo[o].ON_minterms;
            int pichart_words = PInfo[o].pichart_words;
            buffer[t][o].pichart_values = (uint64_t*)calloc((size_t)ON_minterms * (size_t)pichart_words, sizeof(uint64_t));
            buffer[t][o].coverage       = (bool*)    calloc((size_t)ON_minterms * (size_t)ON_minterms, sizeof(bool));
            buffer[t][o].decpos         = (int*)     calloc((size_t)ON_minterms, sizeof(int));
            buffer[t][o].covsum         = (int*)     calloc((size_t)ON_minterms, sizeof(int));
            buffer[t][o].fixed_bits     = (uint64_t*)calloc((size_t)ON_minterms * (size_t)implicant_words, sizeof(uint64_t));
            buffer[t][o].value_bits     = (uint64_t*)calloc((size_t)ON_minterms * (size_t)implicant_words, sizeof(uint64_t));
            buffer[t][o].found          = 0;

            if (
                !buffer[t][o].pichart_values || !buffer[t][o].coverage ||
                !buffer[t][o].decpos || !buffer[t][o].covsum ||
                !buffer[t][o].fixed_bits || !buffer[t][o].value_bits
            ) {
                fprintf(stderr, "Error: buffer per-output alloc failed\n");
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }
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

    double scp_time = 0.0, k_scp_time = 0.0;


    int k;
    for (k = search_level; k <= ninputs; k++) {
        int level_start[noutputs];
        for (int o = 0; o < noutputs; ++o) {
            int start_pi = (k > 1 && PInfo[o].nofpi) ? PInfo[o].nofpi[k - 2] : 0;
            if (start_pi < 0 || start_pi > PInfo[o].foundPI) {
                start_pi = PInfo[o].foundPI;
            }
            level_start[o] = start_pi;
        }

        k_scp_time = 0.0; // reset time for this level
        clock_gettime(CLOCK_MONOTONIC, &startk);

        uint64_t maxtasks = nchoosek(ninputs, k);
        atomic_bool time_up;
        atomic_init(&time_up, false);
        double time_up_elapsed = 0.0;
        atomic_uint_fast64_t last_task_reached;
        atomic_init(&last_task_reached, 0ull);
        uint64_t resume_last_task_local = 0ull;
        uint64_t start_task = 0ull;
        ccubes_mutex state_lock;
        if (!ccubes_mutex_init(&state_lock)) {
            fprintf(stderr, "Error: state lock initialization failed\n");
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            return 1;
        }

        if (RESUME_PATH && HAS_RESUME_LAST_TASK && RESUME_K == k) {
            resume_last_task_local = RESUME_LAST_TASK;
            atomic_store_explicit(&last_task_reached, resume_last_task_local, memory_order_release);
            start_task = resume_last_task_local + 1ull;
            if (start_task > maxtasks) start_task = maxtasks; // safety
        }

        if (maxtasks == 0) {
            // overflow, too many tasks
            ccubes_mutex_destroy(&state_lock);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            // exit(EXIT_FAILURE);
            return(1);
        }

        DBG_INFO_BLOCK {
            fprintf(debug_out, "\nk: %d\n", k);
            fprintf(debug_out, "maxtasks: %lld\n", maxtasks);
        }

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
            .max_shared = &max_shared,
            .increase = increase,
            .multiplier = &multiplier,
            .deterministic_order = DETERMINISTIC_PI_ORDER,
            .time_limit_sec = TIME_LIMIT_SEC,
            .base_elapsed = BASE_ELAPSED,
            .start_time = start,
            .time_up = &time_up,
            .time_up_elapsed = &time_up_elapsed,
            .last_task_reached = &last_task_reached,
            .state_lock = &state_lock
        };

        if (
            !ccubes_parallel_for(
                start_task,
                maxtasks,
                THREADS,
                true,
                pi_search_range_worker,
                &pi_ctx
            )
        ) {
            fprintf(stderr, "Error: failed to start workers for PI search\n");
            ccubes_mutex_destroy(&state_lock);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            return 1;
        }

        HAS_RESUME_LAST_TASK = false; // reset for the next k-level

        // Time-limit checkpoint (post-loop or early-cancel)
        bool level_time_up = atomic_load_explicit(&time_up, memory_order_acquire);
        uint64_t last_task_reached_value = atomic_load_explicit(
            &last_task_reached,
            memory_order_acquire
        );
        ccubes_mutex_destroy(&state_lock);

        if (DETERMINISTIC_PI_ORDER) {
            for (int o = 0; o < noutputs; ++o) {
                if (!canonicalize_pi_order(&PInfo[o], implicant_words, level_start[o])) {
                    fprintf(stderr, "Error: failed to canonicalize PI order for output %d\n", o + 1);
                    destroy_output_locks(output_locks, noutputs);
                    cleanup(PInfo, buffer);
                    return 1;
                }
            }
        }

        if (TIME_LIMIT_SEC > 0 && level_time_up) {
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
            int checkpoint_coverage[noutputs];
            for (int o = 0; o < noutputs; ++o) {
                checkpoint_coverage[o] = certified_states[o].coverage_horizon;
            }

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
                    time_up_elapsed,
                    BASE_SCP + scp_time,
                    last_task_reached_value,
                    CERTIFIED_MODE,
                    checkpoint_coverage
                ) == 0
            ) {
                double pct = (double)last_task_reached_value * 100.0 / (double)maxtasks;

                fprintf(
                    stderr,
                    "Time limit reached. Checkpoint saved at %.2f%% of level k=%d.\n",
                    pct, k
                );
            } else {
                fprintf(
                    stderr,
                    "Time limit reached. Failed to save checkpoint to %s.\n",
                    CHK_SAVE_PATH
                );
            }
            if (tmp_dst_alloc) free(tmp_dst_alloc);
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            return 0;
        }

        // solve the PI charts per output
        if (POOL_MAX == 1) {
            for (int o = 0; o < noutputs; o++) {
                int *foundPI = &PInfo[o].foundPI;
                bool *ON_set_covered = &PInfo[o].ON_set_covered;
                int ON_minterms = PInfo[o].ON_minterms;
                int *pichart = PInfo[o].pichart;
                int *prevsolmin = &PInfo[o].prevsolmin;
                int *solmin = &PInfo[o].solmin;
                int *previndices = PInfo[o].previndices;
                int *indices = PInfo[o].indices;

                PInfo[o].nofpi[k - 1] = *foundPI; // TODO move this after checking the coverage at this level of k

                if (*foundPI > 0 && !*ON_set_covered) {
                    bool test_coverage = true;

                    int r = 0;
                    while (r < ON_minterms && test_coverage) {

                        bool minterm_covered = false;
                        int c = 0;

                        while (c < *foundPI && !minterm_covered) {
                            minterm_covered = pichart[c * ON_minterms + r];
                            c++;
                        }

                        test_coverage = minterm_covered;
                        r++;
                    }

                    *ON_set_covered = test_coverage;
                }

                certified_stop_observe_coverage(
                    &certified_states[o],
                    k,
                    *ON_set_covered
                );


                if (*ON_set_covered && !PInfo[o].stop_search) {
                    bool boundary_exact = SCP_TYPE == 1;
                    DBG_TRACE_BLOCK {
                        fprintf(debug_out, "Output %d, found PIs: %d", o + 1, *foundPI);
                    }

                    double *weights = NULL;

                    if (WEIGHT_PIC > 0) {
                        weights = calloc(*foundPI, sizeof(double));
                        layer_weights[k] = 1;
                        for (int i = k - 1; i > 0; i--) {
                            layer_weights[i] = layer_weights[i + 1] * 2;
                        }

                        int counter = 0;
                        for (int l = 1; l < k; l++) {
                            int layer_pis = PInfo[o].nofpi[l] - PInfo[o].nofpi[l - 1];
                            for (int i = 0; i < layer_pis; i++) {
                                weights[counter] = layer_weights[l];
                                if (WEIGHT_PIC == 2) {
                                    weights[counter] += 1 * (PInfo[o].shared[counter] - 1); // additional weight for shared PIs
                                }
                                counter++;
                            }
                        }
                    }

                    clock_gettime(CLOCK_MONOTONIC, &startg);

                    if (SCP_TYPE == 0) { // Bundled hybrid solver
                        solve_scp_lagrangian(
                            pichart,
                            *foundPI,
                            ON_minterms,
                            weights,
                            indices,
                            solmin,
                            HYBRID_EFFORT_LEVEL
                        );
                        boundary_exact = lagrangian_last_run_proved_optimal();
                        debug_print_lagrangian_stats(o);
                    }

                    if (SCP_TYPE == 1) { // Gurobi: blended multi-objective
                        gurobi_multiobjective(
                            pichart,
                            *foundPI,
                            ON_minterms,
                            weights,
                            indices,
                            solmin
                        );
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

                        if (!blocking_stop_states[o].certification_required) {
                            stop_counter[o] = 0;
                        }

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

                    if (!certified_blocking_observe_plateau(
                        &blocking_stop_states[o],
                        &certified_states[o],
                        REPORT_BLOCKING_DIAGNOSTIC,
                        !CERTIFIED_MODE,
                        plateau_triggered,
                        stderr,
                        o + 1,
                        &PInfo[o],
                        ninputs,
                        k,
                        indices,
                        *solmin,
                        boundary_exact
                    )) {
                        fprintf(stderr, "Error: blocking diagnostic failed for output %d.\n", o + 1);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }
                    PInfo[o].stop_search = certified_stop_policy_decision(
                        certified_states ? &certified_states[o] : NULL,
                        &blocking_stop_states[o],
                        CERTIFIED_MODE,
                        plateau_triggered,
                        k,
                        *solmin,
                        boundary_exact
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
                int *pichart = PInfo[o].pichart;
                int *solmin = &PInfo[o].solmin;
                int *pool_count = &PInfo[o].pool_count;
                int **pool_solutions = PInfo[o].pool_solutions;

                PInfo[o].nofpi[k - 1] = *foundPI;

                if (*foundPI > 0 && !*ON_set_covered) {
                    bool test_coverage = true;

                    int r = 0;
                    while (r < ON_minterms && test_coverage) {

                        bool minterm_covered = false;
                        int c = 0;

                        while (c < *foundPI && !minterm_covered) {
                            minterm_covered = pichart[c * ON_minterms + r];
                            c++;
                        }

                        test_coverage = minterm_covered;
                        r++;
                    }

                    *ON_set_covered = test_coverage;
                }

                certified_stop_observe_coverage(
                    &certified_states[o],
                    k,
                    *ON_set_covered
                );

                if (*ON_set_covered && !PInfo[o].stop_search) {
                    double *weights = NULL;

                    if (WEIGHT_PIC > 0) {
                        weights = calloc(*foundPI, sizeof(double));
                        layer_weights[k] = 1;
                        for (int i = k - 1; i > 0; i--) {
                            layer_weights[i] = layer_weights[i + 1] * 2;
                        }

                        int counter = 0;
                        for (int l = 1; l < k; l++) {
                            int layer_pis = PInfo[o].nofpi[l] - PInfo[o].nofpi[l - 1];
                            for (int i = 0; i < layer_pis; i++) {
                                weights[counter] = layer_weights[l];
                                if (WEIGHT_PIC == 2) {
                                    weights[counter] += 1 * (PInfo[o].shared[counter] - 1); // additional weight for shared PIs
                                }
                                counter++;
                            }
                        }
                    }

                    clock_gettime(CLOCK_MONOTONIC, &startg);

                    if (SCP_TYPE == 0) { // Bundled hybrid solver with solution pool
                        solve_scp_lagrangian_pool(
                            pichart,
                            *foundPI,
                            ON_minterms,
                            weights,
                            POOL_MAX,
                            pool_count,
                            pool_solutions,
                            solmin,
                            HYBRID_EFFORT_LEVEL
                        );
                        pool_boundary_exact[o] = lagrangian_last_run_proved_optimal();
                        debug_print_lagrangian_stats(o);
                    }

                    if (SCP_TYPE == 1) { // Gurobi: solution pool
                        gurobi_solution_pool(
                            pichart,
                            *foundPI,
                            ON_minterms,
                            POOL_MAX,
                            weights,
                            pool_count,
                            pool_solutions,
                            solmin
                        );
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

                    pool_execution_time[o] = execution_time;

                    free(weights);
                }


                for (int i = 0; i < ON_minterms; i++) {
                    PInfo[o].last_index[i] = PInfo[o].k_last_index[i];
                }

            } // end of outputs loop solving SCP


            /*
             * Cross-output selection from solution pools:
             * For each output, pick the solution in its pool whose PIs are most shared
             * with the other outputs' pools.
             */
            int *chosen_idx = (int*)calloc((size_t)noutputs, sizeof(int));
            if (!chosen_idx) {
                fprintf(stderr, "Error: Memory allocation failed for chosen_idx\n");
                destroy_output_locks(output_locks, noutputs);
                cleanup(PInfo, buffer);
                return 1;
            }

            for (int o = 0; o < noutputs; ++o) {
                int pool_count_val = PInfo[o].pool_count;
                int solmin_o = PInfo[o].solmin;

                if (pool_count_val <= 0 || solmin_o <= 0) {
                    chosen_idx[o] = -1;
                    continue;
                }

                int best_score = -1;
                int best_p = 0;

                for (int p = 0; p < pool_count_val; ++p) {
                    int *sol = PInfo[o].pool_solutions[p];
                    int score = 0;

                    for (int j = 0; j < solmin_o; ++j) {
                        int col = sol[j];
                        // For each other output, count at most once per output if this PI appears anywhere in its pool
                        for (int oo = 0; oo < noutputs; ++oo) {
                            if (oo == o) continue;
                            int pc2 = PInfo[oo].pool_count;
                            int solmin_oo = PInfo[oo].solmin;
                            if (pc2 <= 0 || solmin_oo <= 0) continue;

                            bool found_in_oo = false;
                            for (int pp = 0; pp < pc2 && !found_in_oo; ++pp) {
                                int *sol2 = PInfo[oo].pool_solutions[pp];
                                for (int jj = 0; jj < solmin_oo; ++jj) {
                                    int col2 = sol2[jj];
                                    bool eq = true;
                                    for (int w = 0; w < implicant_words; ++w) {
                                        uint64_t pos1 = PInfo[o].implicants_pos[col * implicant_words + w];
                                        uint64_t val1 = PInfo[o].implicants_val[col * implicant_words + w];
                                        uint64_t pos2 = PInfo[oo].implicants_pos[col2 * implicant_words + w];
                                        uint64_t val2 = PInfo[oo].implicants_val[col2 * implicant_words + w];
                                        if (pos1 != pos2 || val1 != val2) {
                                            eq = false;
                                            break;
                                        }
                                    }
                                    if (eq) {
                                        found_in_oo = true;
                                        break;
                                    }
                                }
                            }
                            if (found_in_oo) score++;
                        }
                    }

                    if (score > best_score) {
                        best_score = score;
                        best_p = p;
                    }
                }

                chosen_idx[o] = best_p;
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
                    bool has_choice = !(pool_count_val <= 0 || chosen_idx[o] < 0);
                    if (has_choice) {
                        int *src = PInfo[o].pool_solutions[chosen_idx[o]];
                        for (int i = 0; i < solmin_o; ++i) {
                            indices[i] = src[i];
                        }
                    }

                    DBG_INFO_BLOCK {
                        if (*solmin > 0 && !PInfo[o].stop_search) {
                            fprintf(debug_out, "[pool] Output %d (SCP %.3fs) solution (%d): ", o + 1, pool_execution_time[o], *solmin);
                            for (int i = 0; i < *solmin; i++) {
                                fprintf(debug_out, "%d ", indices[i] + 1);
                            }
                        }
                    }

                    bool plateau_triggered = false;
                    if (*solmin < *prevsolmin) {
                        *prevsolmin = *solmin;
                        for (int i = 0; i < *solmin; i++) {
                            previndices[i] = indices[i];
                        }

                        if (!blocking_stop_states[o].certification_required) {
                            stop_counter[o] = 0;
                        }

                        DBG_INFO_BLOCK {
                            if (!PInfo[o].stop_search) {
                                fprintf(debug_out, "\n");
                            }
                        }
                    } else {
                        for (int i = 0; i < *solmin; i++) {
                            indices[i] = previndices[i];
                        }

                        *solmin = *prevsolmin;
                        stop_counter[o]++;

                        DBG_INFO_BLOCK {
                            if (!PInfo[o].stop_search) {
                                fprintf(debug_out, "%s\n", stop_counter[o] >= STOP_AFTER_EQUALITY ? "-- stopping search" : "");
                            }
                        }

                        plateau_triggered = stop_counter[o] >= STOP_AFTER_EQUALITY;
                    }

                    if (!certified_blocking_observe_plateau(
                        &blocking_stop_states[o],
                        &certified_states[o],
                        REPORT_BLOCKING_DIAGNOSTIC,
                        !CERTIFIED_MODE,
                        plateau_triggered,
                        stderr,
                        o + 1,
                        &PInfo[o],
                        ninputs,
                        k,
                        indices,
                        *solmin,
                        pool_boundary_exact[o]
                    )) {
                        fprintf(stderr, "Error: blocking diagnostic failed for output %d.\n", o + 1);
                        destroy_output_locks(output_locks, noutputs);
                        cleanup(PInfo, buffer);
                        return 1;
                    }
                    PInfo[o].stop_search = certified_stop_policy_decision(
                        certified_states ? &certified_states[o] : NULL,
                        &blocking_stop_states[o],
                        CERTIFIED_MODE,
                        plateau_triggered,
                        k,
                        *solmin,
                        pool_boundary_exact[o]
                    );
                }
            }

            free(chosen_idx);

        } // end of searching for solutions with pooling

        clock_gettime(CLOCK_MONOTONIC, &endk);
        execution_time =
            (endk.tv_sec - startk.tv_sec) +
            (endk.tv_nsec - startk.tv_nsec) / 1e9;

        DBG_INFO_BLOCK {
            fprintf(debug_out, "k-level execution completed in %.3f (%.3f) seconds\n", execution_time, k_scp_time);
        }

        scp_time += k_scp_time;

        bool stop = true;
        for (int o = 0; o < noutputs; o++) {
            stop &= PInfo[o].stop_search;
        }
        if (stop) break;

    } // end of k loop

    DBG_TRACE_BLOCK {
        fprintf(debug_out, "max shared: %d\n", max_shared);
        fprintf(debug_out, "\n--- END ---\n");
    }

    for (int o = 0; o < noutputs; ++o) {
        bool certificate_required =
            CERTIFIED_MODE || blocking_stop_states[o].certification_required;
        if (certificate_required && !PInfo[o].stop_search) {
            fprintf(
                stderr,
                CERTIFIED_MODE
                    ? "Error: no global optimality certificate was established for output %d.\n"
                    : "Error: output %d was escalated by the blocking warning but no global optimality certificate was established.\n",
                o + 1
            );
            destroy_output_locks(output_locks, noutputs);
            cleanup(PInfo, buffer);
            debug_close();
            return 1;
        }
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

    write_pla_file(DST_FILE, PInfo);

    // Free allocated memory
    destroy_output_locks(output_locks, noutputs);
    cleanup(PInfo, buffer);

    // Calculate and log execution time (accumulated across resumes)
    clock_gettime(CLOCK_MONOTONIC, &end);
    execution_time =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    double total_exec_time = BASE_ELAPSED + execution_time;
    double total_scp_time  = BASE_SCP + scp_time;

    fprintf(stderr, "Execution completed in %.3f (%.3f SCP) seconds\n", total_exec_time, total_scp_time);

    DBG_INFO_BLOCK {
        fprintf(debug_out, "all good.\n");
    }

    debug_close();
    return 0;

}
