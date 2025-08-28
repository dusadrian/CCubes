/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "main.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void help() {
    printf("Usage: ccubes [options] source.pla [dest.pla]\n");
    printf("Options:\n");
    printf("  -k<number>          : start searching from level k\n");
    printf("  -e<number>          : end criterion (default +1 level with the same minima)\n");
    printf("  -b<number>          : bits per word, either 8, 16, 32, 64 (default) or 128\n");
    printf("  -c<number>          : number of CPU cores / threads to use, if OpenMP available\n");
    printf("  -w<number>          : weights applied to the prime implicants:\n");
    printf("                           0 no weight\n");
    printf("                           1 (default) weight based on complexity levels k\n");
    printf("                           2 additional weight if shared between outputs\n");
    printf("  -s<number>          : how to solve the covering problem:\n");
    printf("                           0 greedy (fast)\n");
    printf("                           11 (default) Gurobi exact\n");
    printf("                           12 (not implemented), detect the most shared PIs\n");
    printf("  -d<level>[=<file>] : incremental debug information\n");
    printf("                           0 error only\n");
    printf("                           1 errors + warnings\n");
    printf("                           2 errors + warnings + info\n");
    printf("                           3 everything (trace)\n");
    printf("  -h, --help          : show this help message\n");
}

int main(int argc, char *argv[]) {

    bool gurobi_ok = false;
    #ifdef HAVE_GUROBI
        gurobi_ok = true;
    #endif

    // Record start time for execution timing
    struct timespec start, end, startk, endk, startg, endg;
    double execution_time;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // defaults
    int START_LEVEL = 1;
    int MAX_LEVELS = 1;
    int BITS_PER_WORD = 64;
    int THREADS = 0; // max by default, if OpenMP enabled
    int WEIGHT_PIC = 2;
    int SCP_TYPE = 11;
    char *SRC_FILE = NULL;
    char *DST_FILE = NULL;

    if (argc < 2) {
        help();
        return 1;
    }

    // parse arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-k", 2) == 0) {
            START_LEVEL = atoi(argv[i] + 2);
        } else if (strncmp(argv[i], "-e", 2) == 0) {
            MAX_LEVELS = atoi(argv[i] + 2);
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
        } else if (strncmp(argv[i], "-s", 2) == 0) {
            SCP_TYPE = atoi(argv[i] + 2);
            if (SCP_TYPE != 0 && SCP_TYPE != 11) {
                fprintf(stderr, "Invalid SCP type: %d (must be 0 or 11)\n", SCP_TYPE);
                help();
                return 1;
            }
        } else if (strncmp(argv[i], "-c", 2) == 0) {
            THREADS = atoi(argv[i] + 2);
            if (THREADS < 0) THREADS = 0;
        } else if (strncmp(argv[i], "-d", 2) == 0) {
            char *opt = argv[i] + 2;  // string after "-d"
            char *eq  = strchr(opt, '=');

            int lvl   = DBG_ERROR;
            const char *file = NULL;

            if (eq) {
                // split into number + filename
                *eq = '\0';
                lvl = atoi(opt);
                file = eq + 1;
            } else {
                lvl = atoi(opt);
            }

            // validate level (default to ERROR if invalid)
            if (lvl < DBG_ERROR || lvl > DBG_TRACE) {
                fprintf(stderr, "Invalid debug level: %d (must be 0–3)\n", lvl);
                help();
                return 1;
            }

            debug_init(file, lvl);

        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help();
            return 0;
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

    if (SCP_TYPE == 11 || SCP_TYPE == 12) {
        if (!gurobi_ok) {
            SCP_TYPE = 0;
            WEIGHT_PIC = 0;
            printf("Warning: Gurobi not available, falling back to the simple greedy solver.\n");
        }
    }

    if (SCP_TYPE == 12) {
        WEIGHT_PIC = 2;
    }

    if (!SRC_FILE) {
        fprintf(stderr, "Error: source .pla file is required.\n");
        help();
        return 1;
    }

    int max_shared = 0;
    int multiplier = 0;

    DBG_INFO_BLOCK {
        fprintf(debug_out, "--- START ---\n");
        fprintf(debug_out, "SCP_TYPE: %d\n", SCP_TYPE);
    }

    PIstorage *PInfo = NULL;
    int *nofvalues = NULL;
    int ninputs = 0, noutputs = 0, max_value = 0;


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

    int bits_needed = (int)ceil(log2(max_value));
    int value_bit_width = 1; // at least 1 bits is needed to represent binary values
    while (value_bit_width < bits_needed) {
        value_bit_width *= 2; // Round up to the next power of 2
    }

    if (value_bit_width > BITS_PER_WORD) {
        BITS_PER_WORD = value_bit_width; // Adjust the bits per word
    }

    uint64_t VALUE_BIT_MASK = (1ULL << value_bit_width) - 1ULL;

    // Words needed per PI representation
    int implicant_words = (ninputs * value_bit_width + BITS_PER_WORD - 1) / BITS_PER_WORD;



    int increase = 100000; // how much to increase the size of the arrays when needed

    // many PIs will have the same coverage, but they don't necessarily cover the same minterms
    // to employ row dominance when solving the coverage matrix, we need to compare the coverage of
    // the current PI with the coverage of the previous PIs. If this PI survives the comparison, its
    // coverage has to be added in the "covered" vector, at the last index of the PI coverage with
    // the same number of (covered) minterms
    // int last_index[ON_minterms]; // descending order

    int layer_weights[ninputs + 1];
    layer_weights[0] = 0; // k basically starts with 1, so its k-layer weight will start on position 1


    #ifdef _OPENMP
        int num_procs = omp_get_num_procs();
        // In debug mode, force single-thread for safety
        // if (debug_enabled) { // set in debug.c
        //     THREADS = 1;
        // }
        if (THREADS <= 0 || THREADS > num_procs) {
            THREADS = num_procs; // default to max available threads
        }
        omp_set_num_threads(THREADS);
    #endif

    if (THREADS == 0) { // which means no OpenMP was detected
        THREADS = 1;
    }

    // Implement thread-private buffer space
    ThreadBuffer **buffer = (ThreadBuffer**)calloc(THREADS, sizeof(ThreadBuffer*));
    if (!buffer) {
        fprintf(stderr, "Error: buffer allocation failed\n");
        cleanup(PInfo, buffer);
        return 1;
    }

    int stop_counter[noutputs];
    for (int o = 0; o < noutputs; o++) {

        // --- temporary debugging code ---
        // printf("Output %d, ON_minterms: %d, OFF_minterms: %d\n", o + 1, PInfo[o].ON_minterms, PInfo[o].OFF_minterms);
        // --------------------------------

        // the original ON-set was transposed to a column-major matrix
        int ON_minterms = PInfo[o].ON_minterms;

        // preallocating for a moderate number; grows dynamically as needed
        PInfo[o].estimPI = 500000;
        // PInfo[o].k_estimPI = 1000000;

        // the index of the PIs, in descending order of their number of covered ON-set minterms
        PInfo[o].covered = (int *) calloc(PInfo[o].estimPI, sizeof(int));
        if (!PInfo[o].covered) {
            fprintf(stderr, "Error: Memory allocation failed for covered array\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].last_index = (int *) calloc(ON_minterms, sizeof(int));
        PInfo[o].k_last_index = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].last_index || !PInfo[o].k_last_index) {
            fprintf(stderr, "Error: Memory allocation failed for index arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        // prefixing (int *) before calloc() prefills all values sizeof(with) 0s
        PInfo[o].pichart = (int *) calloc(PInfo[o].estimPI * ON_minterms, sizeof(int));
        if (!PInfo[o].pichart) {
            fprintf(stderr, "Error: Memory allocation failed for pichart\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].pichart_words = (ON_minterms + BITS_PER_WORD - 1) / BITS_PER_WORD; // Words needed per coverage matrix columns
        PInfo[o].pichart_pos = (uint64_t *) calloc(PInfo[o].estimPI * PInfo[o].pichart_words, sizeof(uint64_t));
        if (!PInfo[o].pichart_pos) {
            fprintf(stderr, "Error: Memory allocation failed for pichart_pos\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].implicants_pos = (uint64_t *) calloc(PInfo[o].estimPI * implicant_words, sizeof(uint64_t));
        PInfo[o].implicants_val = (uint64_t *) calloc(PInfo[o].estimPI * implicant_words, sizeof(uint64_t));
        if (!PInfo[o].implicants_pos || !PInfo[o].implicants_val) {
            fprintf(stderr, "Error: Memory allocation failed for implicants arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].shared = (int *) calloc(PInfo[o].estimPI, sizeof(int));
        PInfo[o].covsum = (int *) calloc(PInfo[o].estimPI, sizeof(int));
        if (!PInfo[o].shared || !PInfo[o].covsum) {
            fprintf(stderr, "Error: Memory allocation failed for shared/covsum arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].foundPI = 0;
        PInfo[o].solmin = 0;

        // the previous (level k - 1), minimum number of PIs to solve the coverage matrix
        PInfo[o].prevsolmin = ON_minterms + 1; // Initializtion: set to a value larger than the maximum possible number of PIs

        // the positions of the PIs solving the coverage matrix
        // this vector can never be lengthier than the number of ON minterms (ON_minterms)
        PInfo[o].previndices = (int *) calloc(ON_minterms, sizeof(int));
        PInfo[o].indices = (int *) calloc(ON_minterms, sizeof(int));
        if (!PInfo[o].previndices || !PInfo[o].indices) {
            fprintf(stderr, "Error: Memory allocation failed for indices arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].ON_set_covered = false;

        PInfo[o].cov_word_index = (int *) calloc(ON_minterms, sizeof(int));
        PInfo[o].shifted_cov_mask = (uint64_t *) calloc(ON_minterms, sizeof(uint64_t));
        if (!PInfo[o].cov_word_index || !PInfo[o].shifted_cov_mask) {
            fprintf(stderr, "Error: Memory allocation failed for coverage arrays\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        for (int r = 0; r < ON_minterms; r++) {
            PInfo[o].cov_word_index[r] = r / BITS_PER_WORD;
            PInfo[o].shifted_cov_mask[r] = (1ULL << (r % BITS_PER_WORD));
        }

        // store the number of PIs at each level of complexity k
        PInfo[o].nofpi = (int *) calloc(ninputs, sizeof(int));
        if (!PInfo[o].nofpi) {
            fprintf(stderr, "Error: Memory allocation failed for nofpi array\n");
            cleanup(PInfo, buffer);
            return 1;
        }

        PInfo[o].stop_search = ON_minterms == 0; // if there are no ON-set minterms, then stop searching for PIs

    }

    // allocate memory buffer for each thread
    for (int t = 0; t < THREADS; t++) {
        buffer[t] = (ThreadBuffer*)calloc(noutputs, sizeof(ThreadBuffer));
        if (!buffer[t]) {
            fprintf(stderr, "Error: buffer[%d] allocation failed\n", t);
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
        #ifdef _OPENMP
            fprintf(debug_out, "OpenMP enabled, %d workers\n", omp_get_max_threads());
        #endif
    }


    int bit_index[ninputs];  //
    int word_index[ninputs]; //   for the implicants matrix
    uint64_t shifted_mask[ninputs];
    for (int r = 0; r < ninputs; r++) {
        bit_index[r] = (r % (BITS_PER_WORD / value_bit_width)) * value_bit_width;   // Bit position within the word
        word_index[r] = r / (BITS_PER_WORD / value_bit_width);                      // Word index within the implicant
        shifted_mask[r] = (VALUE_BIT_MASK << bit_index[r]);
    }


    int k;
    for (k = START_LEVEL; k <= ninputs; k++) {

        clock_gettime(CLOCK_MONOTONIC, &startk);

        uint64_t maxtasks = nchoosek(ninputs, k);
        if (maxtasks == 0) {
            // overflow, too many tasks
            cleanup(PInfo, buffer);
            // exit(EXIT_FAILURE);
            return(1);
        }

        DBG_INFO_BLOCK {
            fprintf(debug_out, "\nk: %d\n", k);
            fprintf(debug_out, "maxtasks: %lld\n", maxtasks);
        }

        // Parallelize tasks loop with thread-local buffer buffers (see buffer[tid][o])
        #ifdef _OPENMP
            #pragma omp parallel for if(THREADS > 1) schedule(static, 1)
        #endif
        for (uint64_t task = 0; task < maxtasks; task++) {

            int tid = 0;
            #ifdef _OPENMP
                tid = omp_get_thread_num();
            #endif

            int tempk[k];
            uint64_t combination = task;

            DBG_INFO_BLOCK {
                if (task % 1000000 == 0 && task / 1000000 > 0) { // every 1M tasks
                    fprintf(debug_out, "-");
                }
                if (task % 100000000 == 0 && task / 100000000 > 0) { // every 100M tasks
                    fprintf(debug_out, " (%lld)", task / 100000000);
                }
                if (task % 50000000 == 0 && task / 50000000 > 0) { // every 50M tasks
                    fprintf(debug_out, "\n");
                }
            }

            // fill the combination for the current task / combination number
            int x = 0;
            for (int i = 0; i < k; i++) {
                while (1) {
                    uint64_t cval = nchoosek(ninputs - (x + 1), k - (i + 1));
                    if (cval == 0 || cval > combination) break; // guard against overflow/invalid
                    combination -= cval;
                    x++;
                }
                // clamp to valid range [0, ninputs-1]
                if (x < 0) x = 0;
                if (x >= ninputs) x = ninputs - 1;
                tempk[i] = x;
                x++;
            }

            DBG_TRACE_BLOCK {
                fprintf(debug_out, "tempk: ");
                for (int i = 0; i < k; i++) {
                    fprintf(debug_out, "%d ", tempk[i] + 1);
                }
                fprintf(debug_out, "\n");
            }

            uint64_t fixed_bits[implicant_words];
            for (int w = 0; w < implicant_words; w++) {
                fixed_bits[w] = 0ULL;
            }
            for (int c = 0; c < k; c++) {
                fixed_bits[word_index[tempk[c]]] |= shifted_mask[tempk[c]]; // for implicants_pos
            }

            int max_found = 0;
            for (int o = 0; o < noutputs; o++) {
                int ON_minterms = PInfo[o].ON_minterms;

                if (ON_minterms == 0 || PInfo[o].stop_search) {
                    continue;
                }

                int OFF_minterms = PInfo[o].OFF_minterms;
                int *covered = PInfo[o].covered;
                int *last_index = PInfo[o].last_index;
                int pichart_words = PInfo[o].pichart_words;
                uint64_t *pichart_pos = PInfo[o].pichart_pos; // mask for the PI chart values
                int *cov_word_index = PInfo[o].cov_word_index;
                uint64_t *shifted_cov_mask = PInfo[o].shifted_cov_mask;

                ThreadBuffer *ts = &buffer[tid][o];
                uint64_t *task_pichart_values = ts->pichart_values;
                bool *task_coverage = ts->coverage;
                int *task_found = &ts->found;



                // allocate vectors of decimal numbers for the ON-set and OFF-set rows
                int decpos[ON_minterms];
                int decneg[OFF_minterms];

                // create the vector of multiple bases, useful when calculating the decimal representation
                // of a particular combination of columns, for each row
                int mbase[k];
                mbase[0] = 1; // the first number is _always_ equal to 1, irrespective of the number of values in a certain input

                // calculate the vector of multiple bases, for example if we have k = 3 (three inputs) with
                // 2, 3 and 2 values then mbase will be [1, 2, 6] from: 1, 1 * 2 = 2, 2 * 3 = 6
                for (int i = 1; i < k; i++) {
                    mbase[i] = mbase[i - 1] * nofvalues[tempk[i - 1]];
                }

                // calculate decimal numbers, using mbase, fills in decpos and decneg

                int unique_off_rows[OFF_minterms];
                bool dc_off_rows[OFF_minterms];
                int off_count = 0;

                // initialize don't-care flags to false
                for (int r = 0; r < OFF_minterms; r++) {
                    dc_off_rows[r] = false;
                }

                for (int r = 0; r < OFF_minterms; r++) {
                    int acc = 0;
                    bool has_dc = false;

                    for (int c = 0; c < k; c++) {
                        int value = PInfo[o].OFF_set[r * ninputs + tempk[c]];
                        if (value == 0) has_dc = true;
                        acc += value * mbase[c];
                    }

                    decneg[r] = acc;
                    dc_off_rows[r] = has_dc;

                    // uniqueness check
                    bool unique = true;
                    for (int prev = 0; prev < off_count; prev++) {
                        if (decneg[unique_off_rows[prev]] == acc) {
                            unique = false;
                            break;
                        }
                    }
                    if (unique) {
                        unique_off_rows[off_count++] = r;
                    }
                }


                int possible_rows[ON_minterms];
                int found = 0;

                for (int r = 0; r < ON_minterms; r++) {
                    bool valid_row = true;
                    int acc = 0;
                    for (int c = 0; c < k; c++) {
                        int value = PInfo[o].ON_set[r * ninputs + tempk[c]];
                        if (value == 0) {
                            valid_row = false;
                            break;
                        }
                        acc += value * mbase[c];
                    }
                    decpos[r] = acc;

                    if (!valid_row) continue;

                    int prev = 0;
                    // check if the row is unique
                    while (prev < found && valid_row) {
                        valid_row = decpos[possible_rows[prev]] != decpos[r];
                        prev++;
                    }

                    if (!valid_row) continue;

                    // check if the row is different from any OFF-set row
                    for (int roff = 0; roff < off_count; roff++) {
                        bool different = false;
                        if (dc_off_rows[unique_off_rows[roff]]) {
                            for (int c = 0; c < k; c++) {
                                int v_ON = PInfo[o].ON_set[r * ninputs + tempk[c]];
                                int v_OFF = PInfo[o].OFF_set[unique_off_rows[roff] * ninputs + tempk[c]];

                                if (v_OFF != 0 && v_OFF != v_ON) {
                                    different = true;
                                    break;
                                }
                            }
                        } else {
                            different = decpos[r] != decneg[unique_off_rows[roff]];
                        }

                        if (!different) {
                            valid_row = false;
                            break;
                        }
                    }

                    if (!valid_row) continue;

                    possible_rows[found] = r;
                    found++;
                    max_found++;
                }

                for (int f = 0; f < found; f++) {
                    // using bit shifting, store the fixed bits and value bits
                    uint64_t value_bits[implicant_words];

                    for (int w = 0; w < implicant_words; w++) {
                        value_bits[w] = 0ULL;
                    }

                    for (int c = 0; c < k; c++) {
                        int value = PInfo[o].ON_set[possible_rows[f] * ninputs + tempk[c]] - 1;
                        // set the relevant bits
                        value_bits[word_index[tempk[c]]] |= ((uint64_t)value << bit_index[tempk[c]]);
                    }

                    uint64_t pichart_values[pichart_words];
                    for (int w = 0; w < pichart_words; w++) {
                        pichart_values[w] = 0ULL;
                    }

                    bool coverage[ON_minterms];
                    int covsum = 0;
                    for (int r = 0; r < ON_minterms; r++) {
                        coverage[r] = decpos[r] == decpos[possible_rows[f]];
                        if (coverage[r]) {
                            pichart_values[cov_word_index[r]] |= shifted_cov_mask[r];
                            // TODO: store the index of the covered column somewhere
                            // for a greedy algorithm if the SCP is too complex
                            covsum++;
                        }
                    }

                    // check if the current PI is not redundant by row dominance
                    // against the PIs at the previous level of complexity k - 1
                    // last_index does not (yet) contain any PI indexes at the current level of complexity k
                    bool redundant = false;

                    if (covsum > 0) {
                        for (int rd = 0; rd < last_index[covsum - 1]; rd++) {
                            bool dominated = true;
                            for (int w = 0; w < pichart_words; w++) {
                                if ((pichart_values[w] & pichart_pos[covered[rd] * pichart_words + w]) != pichart_values[w]) {
                                    dominated = false;
                                    break;
                                }
                            }

                            if (dominated) {
                                redundant = true;
                                break;
                            }
                        }
                    }

                    if (redundant) continue;

                    // add everything to the temporary / task storage objects

                    for (int w = 0; w < pichart_words; w++) {
                        // the dereference operator in *task_found has precedence
                        // over the multiplication operator *
                        task_pichart_values[*task_found * pichart_words + w] = pichart_values[w];
                    }

                    for (int r = 0; r < ON_minterms; r++) {
                        task_coverage[*task_found * ON_minterms + r] = coverage[r];
                    }

                    ts->decpos[*task_found] = decpos[possible_rows[f]];
                    ts->covsum[*task_found] = covsum;

                    for (int w = 0; w < implicant_words; w++) {
                        ts->fixed_bits[*task_found * implicant_words + w] = fixed_bits[w];
                        ts->value_bits[*task_found * implicant_words + w] = value_bits[w];
                    }

                    (*task_found)++;

                } // end of found loop
            } // end of outputs loop

            // Identify unique PIs across all outputs, and determine which are shared
            // across multiple outputs. This is done by creating a map of unique PIs
            // and counting how many outputs each unique PI belongs to.

            if (max_found > 0) {
                // matrices with max_found columns and noutputs rows
                int *output_map = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int));
                int *covsum_map = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int));
                int *found_map  = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int)); // the f index within the output vector

                // single vectors
                int *uniquePIs    = (int *) calloc((size_t)max_found, sizeof(int));
                int *shared_count = (int *) calloc((size_t)max_found, sizeof(int));

                int counter = 0; // counter for the unique PIs

                for (int o = 0; o < noutputs; o++) {
                DBG_TRACE_BLOCK {
                    fprintf(debug_out, "Output %d, found PIs:", o + 1);
                }
                for (int f = 0; f < buffer[tid][o].found; f++) {
                    DBG_TRACE_BLOCK {
                        fprintf(debug_out, " %d", buffer[tid][o].decpos[f]);
                    }
                    bool unique = true;

                    for (int u = 0; u < counter; u++) {
                        if (buffer[tid][o].decpos[f] == uniquePIs[u]) {
                            output_map[u * noutputs + shared_count[u]] = o;
                            found_map[u * noutputs + shared_count[u]] = f;
                            covsum_map[u * noutputs + shared_count[u]] = buffer[tid][o].covsum[f];
                            shared_count[u]++;
                            unique = false;
                            break;
                        }
                    }

                    if (unique) {
                        uniquePIs[counter] = buffer[tid][o].decpos[f];
                        output_map[counter * noutputs + 0] = o;
                        found_map[counter * noutputs + 0] = f;
                        covsum_map[counter * noutputs + 0] = buffer[tid][o].covsum[f];
                        shared_count[counter]++;
                        counter++;
                    }
                }
                DBG_TRACE_BLOCK {
                    fprintf(debug_out, "\n");
                }
            }

            DBG_TRACE_BLOCK {
                // for (int o = 0; o < noutputs; o++) {
                //     fprintf(debug_out, "Output %d, found PIs: %d\n", o + 1, PInfo[o].task_found);
                //     for (int f = 0; f < PInfo[o].task_found; f++) {
                //         fprintf(debug_out, "  PI %d: decpos = %d, covsum = %d\n",
                //                 f + 1,
                //                 PInfo[o].task_decpos[f],
                //                 PInfo[o].task_covsum[f]);
                //         fprintf(debug_out, "    fixed bits: ");
                //         for (int w = 0; w < implicant_words; w++) {
                //             fprintf(debug_out, "%llu ", PInfo[o].task_fixed_bits[f * implicant_words + w]);
                //         }
                //         fprintf(debug_out, "\n    value bits: ");
                //         for (int w = 0; w < implicant_words; w++) {
                //             fprintf(debug_out, "%llu ", PInfo[o].task_value_bits[f * implicant_words + w]);
                //         }
                //         fprintf(debug_out, "\n");
                //     }
                // }

                // for (int u = 0; u < counter; u++) {
                //     fprintf(debug_out, "Unique PI %d: decpos = %d, shared_count = %d, outputs: ",
                //             u + 1,
                //             uniquePIs[u],
                //             shared_count[u]);
                //     for (int s = 0; s < shared_count[u]; s++) {
                //         fprintf(debug_out, "%d ", output_map[u * max_found + s] + 1);
                //     }
                //     fprintf(debug_out, "\n");
                // }
            }

            for (int u = 0; u < counter; u++) {
                for (int s = 0; s < shared_count[u]; s++) {
                    int f = found_map[u * noutputs + s];
                    int u_covsum = covsum_map[u * noutputs + s];

                    // the output this PI belongs to
                    int o = output_map[u * noutputs + s];

                    int ON_minterms = PInfo[o].ON_minterms;
                    int *covered = PInfo[o].covered;
                    int *last_index = PInfo[o].last_index;
                    int *k_last_index = PInfo[o].k_last_index;
                    int pichart_words = PInfo[o].pichart_words;
                    uint64_t *pichart_pos = PInfo[o].pichart_pos;
                    int *pichart = PInfo[o].pichart;
                    uint64_t *implicants_pos = PInfo[o].implicants_pos;
                    uint64_t *implicants_val = PInfo[o].implicants_val;
                    int *estimPI = &PInfo[o].estimPI;
                    int *foundPI = &PInfo[o].foundPI;
                    int *shared = PInfo[o].shared;
                    int *covsum = PInfo[o].covsum;

                    uint64_t *task_pichart_values = buffer[tid][o].pichart_values;

                    bool redundant = false;

                    // check if the PI is redundant by row dominance
                    // but now against the previous PIs from the SAME complexity level k

                    int start_index = (k == 1 || u_covsum <= 1) ? 0 : last_index[u_covsum - 1];

                    for (int rd = start_index; rd < ((u_covsum <= 1) ? 0 : k_last_index[u_covsum - 1]); rd++) {
                        bool dominated = true;
                        for (int w = 0; w < pichart_words; w++) {
                            if ((task_pichart_values[f * pichart_words + w] & pichart_pos[covered[rd] * pichart_words + w]) != task_pichart_values[f * pichart_words + w]) {
                                dominated = false;
                                break;
                            }
                        }

                        if (dominated) {
                            redundant = true;
                            break;
                        }
                    }

                    if (redundant) continue;

                    // Sanitize covsum bounds to avoid OOB on k_last_index
                    if (u_covsum < 1) u_covsum = 1;
                    if (u_covsum > ON_minterms) u_covsum = ON_minterms;

                    #ifdef _OPENMP
                        #pragma omp critical
                    #endif
                    {

                        // Ensure capacity before writing the next PI
                        if ((*foundPI + 1) > *estimPI) {
                            resize((void**)&pichart,        TYPE_INT,    increase, *estimPI, ON_minterms);
                            resize((void**)&pichart_pos,    TYPE_UINT64, increase, *estimPI, pichart_words);
                            resize((void**)&implicants_pos, TYPE_UINT64, increase, *estimPI, implicant_words);
                            resize((void**)&implicants_val, TYPE_UINT64, increase, *estimPI, implicant_words);
                            resize((void**)&shared,         TYPE_INT,    increase, *estimPI, 1);
                            resize((void**)&covsum,         TYPE_INT,    increase, *estimPI, 1);
                            resize((void**)&covered,        TYPE_INT,    increase, *estimPI, 1);

                            // Update the PInfo structure pointers after resize
                            PInfo[o].pichart = pichart;
                            PInfo[o].pichart_pos = pichart_pos;
                            PInfo[o].implicants_pos = implicants_pos;
                            PInfo[o].implicants_val = implicants_val;
                            PInfo[o].shared = shared;
                            PInfo[o].covsum = covsum;
                            PInfo[o].covered = covered;

                            *estimPI += increase;

                            DBG_INFO_BLOCK {
                                multiplier++;
                                printf("%dx", multiplier);
                            }
                        }

                        // push the PI information to the global arrays

                        for (int w = 0; w < implicant_words; w++) {
                            implicants_pos[(*foundPI) * implicant_words + w] = fixed_bits[w];
                            implicants_val[(*foundPI) * implicant_words + w] = buffer[tid][o].value_bits[f * implicant_words + w];
                        }

                        // populate the coverage matrix
                        for (int r = 0; r < ON_minterms; r++) {
                            for (int w = 0; w < pichart_words; w++) {
                                pichart_pos[(*foundPI) * pichart_words + w] = buffer[tid][o].pichart_values[f * pichart_words + w];
                            }

                            pichart[(*foundPI) * ON_minterms + r] = buffer[tid][o].coverage[f * ON_minterms + r];
                        }

                        shared[*foundPI] = shared_count[u] - 1;
                        if (max_shared < shared[*foundPI]) {
                            max_shared = shared[*foundPI];
                        }
                        covsum[*foundPI] = u_covsum;

                        int insert_at = k_last_index[u_covsum - 1];
                        if (insert_at < 0) insert_at = 0;
                        if (insert_at > *foundPI) insert_at = *foundPI;

                        for (int i = *foundPI; i > insert_at; i--) {
                            covered[i] = covered[i - 1];
                        }

                        covered[insert_at] = *foundPI;

                        // Shift boundaries for all buckets at or above this covsum
                        for (int l = u_covsum - 1; l < ON_minterms; l++) {
                            k_last_index[l] += 1;
                        }

                        (*foundPI)++;
                    }
                }
            }

                free(output_map);
                free(covsum_map);
                free(found_map);
                free(uniquePIs);
                free(shared_count);
            }

            // reset temporary task objects
            for (int o = 0; o < noutputs; o++) {
                // Only reset the logical count; we overwrite used slots on the next iteration.
                buffer[tid][o].found = 0;
            }

        } // end of tasks loop


        // solve the PI charts per output
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
                DBG_INFO_BLOCK {
                    fprintf(
                        debug_out,
                        "Trying to cover output %d with %d PIs\n",
                        o + 1, *foundPI
                    );
                }
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

            DBG_INFO_BLOCK {
                fprintf(debug_out, "Output %d, found PIs: %d", o + 1, *foundPI);
            }

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

                if (SCP_TYPE == 0) { // simple greedy algorithm

                    solve_scp_greedy(
                        pichart,
                        *foundPI,
                        ON_minterms,
                        weights,
                        indices,
                        solmin
                    );
                }

                if (SCP_TYPE == 11) { // Gurobi: blended multi-objective
                    gurobi_multiobjective(
                        pichart,
                        *foundPI,
                        ON_minterms,
                        weights,
                        indices,
                        solmin
                    );
                }

                if (SCP_TYPE == 12) { // Gurobi: solution pool (incompletely implemented)

                    // The goal is to find as many alternative solutions, containing as many shared PIs as possible
                    // and the best solution would be the one with the greatest number of shared PIs.

                    int solcount = 0;

                    // TODO replace 100 with something related to the number of shared PIs (alternative solutions)
                    int max_solutions = 100;
                    int *solmat = (int *)malloc(max_solutions * ON_minterms * sizeof(int));
                    if (!solmat) {
                        fprintf(stderr, "Error: Memory allocation failed for solution matrix\n");
                        cleanup(PInfo, buffer);
                        return 1;
                    }

                    int *pichart_data = calloc(*foundPI * ON_minterms, sizeof(int));
                    for (int i = 0; i < *foundPI * ON_minterms; i++) {
                        pichart_data[i] = pichart[i];
                    }

                    gurobi_solution_pool(
                        pichart_data,
                        *foundPI,
                        ON_minterms,
                        max_solutions,
                        weights,
                        &solcount,
                        solmat,
                        solmin
                    );

                    // TODO: code to find the best solution

                    free(solmat);
                    free(pichart_data);
                }


                if (*solmin == 0) {
                    fprintf(stderr, "Error: solving the minterm coverage failed.\n");
                    cleanup(PInfo, buffer);
                    return 1;
                }

                clock_gettime(CLOCK_MONOTONIC, &endg);
                execution_time =
                    (endg.tv_sec - startg.tv_sec) +
                    (endg.tv_nsec - startg.tv_nsec) / 1e9;

                DBG_INFO_BLOCK {
                    fprintf(debug_out, ", PI chart solved in %.3f seconds\n", execution_time);
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


                if (*solmin < *prevsolmin) {
                    // either solmin is smaller than the previously found solmin,
                    // or it is the very first time a solmin was found

                    *prevsolmin = *solmin;
                    for (int i = 0; i < *solmin; i++) {
                        previndices[i] = indices[i];
                    }

                    stop_counter[o] = 0;

                } else {
                    // the minimum number of PIs did not change in the current level of complexity
                    // we can safely retain the less complex PIs from the previous level
                    for (int i = 0; i < *solmin; i++) {
                        indices[i] = previndices[i];
                    }

                    *solmin = *prevsolmin;
                    stop_counter[o]++;
                    PInfo[o].stop_search = stop_counter[o] == MAX_LEVELS;
                }

                DBG_INFO_BLOCK {
                    fprintf(debug_out, "solmin: %d%s\n", *solmin, PInfo[o].stop_search ? ", stopping search" : "");
                    for (int i = 0; i < *solmin; i++) {
                        fprintf(debug_out, "%d ", indices[i] + 1);
                    }
                    fprintf(debug_out, "\n");

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

            DBG_INFO_BLOCK {
                fprintf(debug_out, "\n");
            }

            for (int i = 0; i < ON_minterms; i++) {
                PInfo[o].last_index[i] = PInfo[o].k_last_index[i];
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &endk);
        execution_time =
            (endk.tv_sec - startk.tv_sec) +
            (endk.tv_nsec - startk.tv_nsec) / 1e9;

        DBG_INFO_BLOCK {
            fprintf(debug_out, "k-level execution completed in %.3f seconds\n", execution_time);
        }

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


    for (int o = 0; o < noutputs; o++) {
        int solmin = PInfo[o].solmin > 0 ? PInfo[o].solmin : 1; // at least one solution is needed
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
        const char *prefix = "ccubes_";
        char *filename = prefix_basename(SRC_FILE, prefix);
        if (!filename) {
            DST_FILE = "outfile.pla";
        } else {
            DST_FILE = filename;
            // Note: filename memory will be cleaned up along with other resources
        }
    }

    write_pla_file(DST_FILE, PInfo);


    // Free allocated memory
    cleanup(PInfo, buffer);

    // Calculate and log execution time
    clock_gettime(CLOCK_MONOTONIC, &end);
    execution_time =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    fprintf(stderr, "Execution completed in %.3f seconds\n", execution_time);

    DBG_INFO_BLOCK {
        fprintf(debug_out, "all good.\n");
    }

    debug_close();
    return 0;

}
