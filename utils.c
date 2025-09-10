/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "utils.h"
#include "checkpoint.h"

void error_message(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

// Use 128-bit unsigned integers if available (GCC/Clang on 64-bit platforms)
#if defined(__GNUC__) && defined(__SIZEOF_INT128__)
    typedef __uint128_t big_uint;
    #define BIG_UINT_AVAILABLE 1
#else
    #define BIG_UINT_AVAILABLE 0
#endif

/**
 * Compute the binomial coefficient "n choose k" (nCk) with overflow protection.
 *
 * If __uint128_t is available, performs 128-bit intermediate multiplication to
 * support very large inputs, returning 0 if the result exceeds 64-bit capacity.
 *
 * If only 64-bit integers are available, performs step-by-step overflow checks
 * and also returns 0 on any overflow risk.
 */
uint64_t nchoosek(int n, int k) {
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }

    if (k == 0 || k == n) {
        return 1;
    }
    // now k is always > 0

    // Take advantage of symmetry: C(n, k) == C(n, n-k)
    if (k > n - k) {
        k = n - k;
    }

#if BIG_UINT_AVAILABLE

    big_uint result = 1;

    for (int i = 0; i < k; i++) {
        /**
         * Safe unsigned math: result *= (n - i) / (i + 1)
         *
         * Cast operands *before* the subtraction/addition to
         * avoid any signed-to-unsigned conversion that triggers
         * -Wsign-conversion warnings.
         *
         * The loop guarantees that:
         * 0 <= i < k <= n - k < n
         */
        result *= (big_uint)n - (big_uint)i;
        result /= (big_uint)i + (big_uint)1;
    }

    // Check if final result exceeds uint64_t capacity
    if (result > (big_uint)ULLONG_MAX) {
        return 0;
    }

    return (uint64_t)result;

#else

    uint64_t result = 1;

    for (int i = 0; i < k; i++) {
        int diff = n - i;
        if (diff <= 0 || result > ULLONG_MAX / (uint64_t)diff) {
            return 0; // multiplication would overflow
        }
        result *= (uint64_t)diff;

        int denom = i + 1;
        if (denom <= 0 || result % (uint64_t)denom != 0) {
            return 0; // division would be imprecise or overflow
        }
        result /= (uint64_t)denom;
    }

    return result;

#endif
}


void resize(
    void **array,
    ArrayType type,
    int increase,
    int size,
    int nrows
) {
    // Input validation
    if (!array) {
        error_message("NULL array pointer passed to resize.");
    }
    if (type < TYPE_BOOL || type > TYPE_DOUBLE) {
        error_message("Invalid type for resizing.");
    }
    if (increase <= 0 || size < 0 || nrows <= 0) {
        error_message("Invalid parameters for resizing.");
    }

    // Check for overflow in size calculations
    if ((size_t)size > SIZE_MAX / (size_t)nrows ||
        (size_t)(size + increase) > SIZE_MAX / (size_t)nrows) {
        error_message("Size overflow in resize operation.");
    }

    size_t oldsize = (size_t)size * (size_t)nrows;
    size_t newsize = (size_t)(size + increase) * (size_t)nrows;

    // Additional overflow check for element size multiplication
    size_t element_size = 0;
    switch (type) {
        case TYPE_BOOL:
            element_size = sizeof(bool);
            break;
        case TYPE_INT:
        case TYPE_INT_ONES:
            element_size = sizeof(int);
            break;
        case TYPE_UINT64:
            element_size = sizeof(uint64_t);
            break;
        case TYPE_DOUBLE:
            element_size = sizeof(double);
            break;
    }

    if (newsize > SIZE_MAX / element_size) {
        error_message("Memory requirement too large for resize operation.");
    }

    void *tmp = NULL;

    switch (type) {
        case TYPE_BOOL:
            tmp = calloc(newsize, sizeof(bool));
            break;
        case TYPE_INT:
        case TYPE_INT_ONES:
            tmp = calloc(newsize, sizeof(int));
            break;
        case TYPE_UINT64:
            tmp = calloc(newsize, sizeof(uint64_t));
            break;
        case TYPE_DOUBLE:
            tmp = calloc(newsize, sizeof(double));
            break;
    }

    if (tmp == NULL) {
        error_message("Memory allocation failed during resize.");
    }

    if (type == TYPE_INT_ONES) {
        for (size_t i = 0; i < newsize; i++) {
            ((int *) tmp)[i] = 1; // Initialize all elements to 1
        }
    }

    if (*array != NULL) {
        switch (type) {
            case TYPE_BOOL:
                memcpy(tmp, *array, oldsize * sizeof(bool));
                break;
            case TYPE_INT:
            case TYPE_INT_ONES:
                memcpy(tmp, *array, oldsize * sizeof(int));
                break;
            case TYPE_UINT64:
                memcpy(tmp, *array, oldsize * sizeof(uint64_t));
                break;
            case TYPE_DOUBLE:
                memcpy(tmp, *array, oldsize * sizeof(double));
                break;
        }
        free(*array);
    }

    *array = tmp;
}


void trim_whitespace(char *str) {
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str)) str++;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end + 1) = '\0';
}

void read_pla_file(
    const char *filename,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int **nofvalues,
    int *max_value
) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        return;
    }

    char line[1024];
    int inputs = 0, outputs = 0;

    bool has_inputs = false;
    bool has_outputs = false;

    *max_value = 0; // Initialize max_value

    bool has_type = false;
    bool correct_type = false;
    int *ON_minterms = NULL;
    int *OFF_minterms = NULL;

    // First pass: Determine dimensions and count rows
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, ".i ", 3) == 0) {
            inputs = atoi(line + 3);
            has_inputs = true;
        } else if (strncmp(line, ".o ", 3) == 0) {
            outputs = atoi(line + 3);
            has_outputs = true;// Allocate ON_minterms and OFF_minterms
            ON_minterms = (int *)calloc(outputs, sizeof(int));
            OFF_minterms = (int *)calloc(outputs, sizeof(int));

            if (!ON_minterms || !OFF_minterms) {
                printf("Error: Memory allocation failed for ON_minterms or OFF_minterms\n");
                fclose(file);
                return;
            }
        } else if (strncmp(line, ".type ", 6) == 0) {
            trim_whitespace(line);
            has_type = true;
            if (strcmp(line + 6, "fr") == 0) {
                correct_type = true;
            } else {
                printf("Error: Only .type fr PLA files are supported (found: %s)\n", line + 6);
                fclose(file);
                return;
            }
            continue;
        } else if (
            line[0] == '#' ||
            strlen(line) <= 1 ||
            strncmp(line, ".p ", 3) == 0 ||
            strncmp(line, ".e", 2) == 0 ||
            strncmp(line, ".ilb ", 5) == 0 ||
            strncmp(line, ".ob ", 4) == 0
        ) {
            continue;
        } else {

            if (!has_inputs || !has_outputs) {
                printf("Error: Missing .i or .o headers in the .pla file\n");
                fclose(file);
                return;
            }

            char *input_part = strtok(line, " |");
            char *output_part = strtok(NULL, " |");

            if (input_part) {
                trim_whitespace(input_part);
            }

            if (output_part) {
                trim_whitespace(output_part);
            };

            if (
                input_part &&
                output_part &&
                (int)strlen(input_part) == inputs &&
                (int)strlen(output_part) == outputs
            ) {
                for (int i = 0; i < outputs; i++) {
                    if (output_part[i] == '1') {
                        ON_minterms[i]++;
                    } else if (output_part[i] == '0') {
                        OFF_minterms[i]++;
                    }
                }
            }
        }
    }

    if (!has_type || !correct_type) {
        printf("Error: Missing or unsupported .type directive (expected .type fr)\n");
        fclose(file);
        return;
    }

    *ninputs = inputs;
    *noutputs = outputs;

    // Allocate and zero-initialize PIstorage array to ensure all fields start as NULL/0
    *PInfo = (PIstorage *)calloc((size_t)outputs, sizeof(PIstorage));
    if (!*PInfo) {
        printf("Error: Memory allocation failed for PInfo\n");
        free(ON_minterms);
        free(OFF_minterms);
        fclose(file);
        return;
    }

    // Pointers are already NULL due to calloc; explicitly set only those we immediately use below
    for (int o = 0; o < outputs; o++) {
        (*PInfo)[o].ON_set = NULL;
        (*PInfo)[o].OFF_set = NULL;
    }

    int temp_poscols[outputs];
    int temp_negcols[outputs];

    for (int o = 0; o < outputs; o++) {
        (*PInfo)[o].inputs = inputs;
        (*PInfo)[o].outputs = outputs;

        // printf("input %d: ON_minterms=%d, OFF_minterms=%d\n", o, ON_minterms[o], OFF_minterms[o]);
        (*PInfo)[o].ON_minterms = ON_minterms[o];
        (*PInfo)[o].ON_set = (int *)calloc(ON_minterms[o] * inputs, sizeof(int));
        if (!(*PInfo)[o].ON_set && ON_minterms[o] > 0) {
            printf("Error: Memory allocation failed for ON_set[%d]\n", o);
            // Free up previously allocated memory
            for (int c = 0; c <= o; c++) {
                free((*PInfo)[c].ON_set);
                free((*PInfo)[c].OFF_set);
            }
            free(*PInfo);
            free(ON_minterms);
            free(OFF_minterms);
            fclose(file);
            return;
        }

        (*PInfo)[o].OFF_minterms = OFF_minterms[o];
        (*PInfo)[o].OFF_set = (int *)calloc(OFF_minterms[o] * inputs, sizeof(int));
        if (!(*PInfo)[o].OFF_set && OFF_minterms[o] > 0) {
            printf("Error: Memory allocation failed for OFF_set[%d]\n", o);
            // Free up previously allocated memory
            for (int c = 0; c <= o; c++) {
                free((*PInfo)[c].ON_set);
                free((*PInfo)[c].OFF_set);
            }
            free(*PInfo);
            free(ON_minterms);
            free(OFF_minterms);
            fclose(file);
            return;
        }

        temp_poscols[o] = ON_minterms[o];
        temp_negcols[o] = OFF_minterms[o];
    }

    *nofvalues = (int *)calloc(inputs, sizeof(int));
    if (!*nofvalues) {
        printf("Error: Memory allocation failed for nofvalues\n");
        for (int o = 0; o < outputs; o++) {
            free((*PInfo)[o].ON_set);
            free((*PInfo)[o].OFF_set);
        }
        free(*PInfo);
        free(ON_minterms);
        free(OFF_minterms);
        fclose(file);
        return;
    }

    // Second pass: Fill ON_set and OFF_set and calculate max_value
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) <= 1 || strncmp(line, ".i", 2) == 0 || strncmp(line, ".o", 2) == 0 || strncmp(line, ".e", 2) == 0 || strncmp(line, ".p", 2) == 0) continue;

        char *input_part = strtok(line, " |");
        char *output_part = strtok(NULL, " |");

        if (input_part) trim_whitespace(input_part);
        if (output_part) trim_whitespace(output_part);

        if (
            input_part &&
            output_part &&
            (int)strlen(input_part) == inputs &&
            (int)strlen(output_part) == outputs
        ) {

            for (int o = 0; o < outputs; o++) {
                int *target_data = NULL;
                int ncols = 0;
                int col_index = 0;

                if (output_part[o] == '1') {
                    target_data = (*PInfo)[o].ON_set;
                    ncols = ON_minterms[o];
                    col_index = ncols - temp_poscols[o];
                    temp_poscols[o]--;
                } else if (output_part[o] == '0') {
                    target_data = (*PInfo)[o].OFF_set;
                    ncols = (*PInfo)[o].OFF_minterms;
                    col_index = ncols - temp_negcols[o];
                    temp_negcols[o]--;
                } else {
                    continue;
                }

                if (col_index < 0 || col_index >= ncols) {
                    error_message("Invalid col_index.");
                }

                for (int j = 0; j < inputs; j++) {
                    int value = (input_part[j] == '1') ? 2
                            : (input_part[j] == '0') ? 1
                            : 0;
                    target_data[col_index * inputs + j] = value;

                    if (value > *max_value) {
                        *max_value = value;
                    }
                    if (value + 1 > (*nofvalues)[j]) {
                        (*nofvalues)[j] = value + 1;
                    }
                }
            }
        }
    }

    free(ON_minterms);
    free(OFF_minterms);

    fclose(file);
}

void write_pla_file(
    const char *filename,
    PIstorage *PInfo
) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Unable to open file %s for writing\n", filename);
        return;
    }

    int ninputs = PInfo[0].inputs;
    int noutputs = PInfo[0].outputs;

    // Write the header
    fprintf(file, ".i %d\n", ninputs); // Number of inputs
    fprintf(file, ".o %d\n", noutputs); // Number of outputs

    // --- temporary debugging code ---
    // print each solution matrix
    // for (int o = 7; o < 8; o++) {
    //     printf("Output %d: solmin = %d\n", o + 1, PInfo[o].solmin);
    //     for (int r = 0; r < PInfo[o].solmin; r++) {
    //         printf("Row %d: ", r);
    //         for (int c = 0; c < ninputs; c++) {
    //             printf("%d ", PInfo[o].solution[c * PInfo[o].solmin + r]);
    //         }
    //         printf("\n");
    //     }
    // }
    // --------------------------------

    // Data structure to store unique rows
    typedef struct {
        int *inputs; // Input part of the row
        int *outputs; // Output part of the row
    } UniqueRow;

    UniqueRow *unique_rows = NULL;
    int unique_count = 0;

    // Iterate through all outputs and rows to find unique rows
    for (int o = 0; o < noutputs; o++) {
        for (int r = 0; r < PInfo[o].solmin; r++) {
            // Extract the input part of the row from the "solution" matrix
            int *current_row = (int *)malloc(ninputs * sizeof(int));
            if (!current_row) {
                printf("Error: Memory allocation failed for current_row\n");
                return;
            }

            // printf("Processing output %d, row %d: ", o, r);
            for (int c = 0; c < ninputs; c++) {
                current_row[c] = PInfo[o].solution[c * PInfo[o].solmin + r];
                // printf(" %d", current_row[c]);
            }


            // Check if the row is already in unique_rows
            bool is_unique = true;
            for (int i = 0; i < unique_count; i++) {
                if (memcmp(unique_rows[i].inputs, current_row, ninputs * sizeof(int)) == 0) {
                    // Row already exists, update the output part
                    unique_rows[i].outputs[o] = 1;
                    is_unique = false;
                    break;
                }
            }

            // printf(" (%s)\n", is_unique ? "Unique" : "Not unique");

            if (is_unique) {
                // Add the new unique row
                unique_rows = (UniqueRow *)realloc(unique_rows, (unique_count + 1) * sizeof(UniqueRow));
                if (!unique_rows) {
                    printf("Error: Memory allocation failed for unique_rows\n");
                    free(current_row);
                    return;
                }
                unique_rows[unique_count].inputs = current_row;
                unique_rows[unique_count].outputs = (int *)calloc(noutputs, sizeof(int));
                if (!unique_rows[unique_count].outputs) {
                    printf("Error: Memory allocation failed for outputs\n");
                    free(current_row);
                    return;
                }
                unique_rows[unique_count].outputs[o] = 1;
                unique_count++;
            } else {
                free(current_row); // Free memory if the row is not unique
            }
        }
    }

    // Write the total number of unique rows
    fprintf(file, ".p %d\n", unique_count);

    // Write the body
    for (int i = 0; i < unique_count; i++) {
        // Write the input part
        for (int c = 0; c < ninputs; c++) {
            int value = unique_rows[i].inputs[c];
            if (value == 0) {
                fprintf(file, "-"); // Don't care
            } else if (value == 1) {
                fprintf(file, "0"); // Closed gate
            } else if (value == 2) {
                fprintf(file, "1"); // Open gate
            } else {
                fprintf(file, "%d", value - 1); // Multi-value (adjusted)
            }
        }

        fprintf(file, " "); // Separator between input and output

        // Write the output part
        for (int o = 0; o < noutputs; o++) {
            fprintf(file, "%d", unique_rows[i].outputs[o]);
        }

        fprintf(file, "\n"); // End of row
    }

    // Write the end marker
    fprintf(file, ".e\n");

    // Free allocated memory
    for (int i = 0; i < unique_count; i++) {
        free(unique_rows[i].inputs);
        free(unique_rows[i].outputs);
    }
    free(unique_rows);

    fclose(file);
}

void cleanup(PIstorage *PInfo, ThreadBuffer **buffer) {
    int noutputs = PInfo[0].outputs;
    for (int o = 0; o < noutputs; o++) {
        free(PInfo[o].ON_set);
        free(PInfo[o].OFF_set);
        free(PInfo[o].covered);
        free(PInfo[o].last_index);
        free(PInfo[o].k_last_index);
        free(PInfo[o].pichart);
        free(PInfo[o].pichart_pos);
        free(PInfo[o].implicants_pos);
        free(PInfo[o].implicants_val);
        free(PInfo[o].shared);
        free(PInfo[o].covsum);
        free(PInfo[o].previndices);
        free(PInfo[o].indices);
        free(PInfo[o].cov_word_index);
        free(PInfo[o].shifted_cov_mask);
        free(PInfo[o].nofpi);

        free(PInfo[o].solution);
        for (int p = 0; p < PInfo[o].pool_count; p++) {
            free(PInfo[o].pool_solutions[p]);
        }
        free(PInfo[o].pool_solutions);
    }

    free(PInfo);

    int threads = 1;
    if (buffer[0]) {
        threads = buffer[0]->threads;
    }

    // Free buffer buffers
    for (int t = 0; t < threads; t++) {
        if (!buffer[t]) continue;
        for (int o = 0; o < noutputs; o++) {
            free(buffer[t][o].pichart_values);
            free(buffer[t][o].coverage);
            free(buffer[t][o].decpos);
            free(buffer[t][o].covsum);
            free(buffer[t][o].fixed_bits);
            free(buffer[t][o].value_bits);
        }
        free(buffer[t]);
    }
    if (buffer) free(buffer);
}

char *prefix_basename(const char *filepath, const char *prefix) {
    const char *basename = strrchr(filepath, '/');
    if (basename) {
        basename++; // skip past '/'
    } else {
        basename = filepath; // no '/' found
    }

    size_t prefix_len = strlen(prefix);
    size_t base_len = strlen(basename);

    char *new_name = malloc(prefix_len + base_len + 1);
    if (!new_name) return NULL;

    strcpy(new_name, prefix);
    strcat(new_name, basename);

    return new_name;
}

void print_info(const char *INFO_PATH, const int info_level) {
    PIstorage *pi_tmp = NULL;
    int ni=0, no=0, bpw=0, vbw=0, ipw=0, ck=0, ml=0, wp=0, st=0, pm=0, sl=0;
    int *stopc_tmp = NULL; int *nofvals_tmp = NULL; char *src_saved=NULL; char *dst_saved=NULL;
    double elapsed_total = 0.0, elapsed_scp = 0.0; uint64_t last_task = 0ull;

    if (load_checkpoint(
            INFO_PATH,
            &pi_tmp,
            &ni,
            &no,
            &bpw,
            &vbw,
            &ipw,
            &ck,
            &stopc_tmp,
            &ml,
            &wp,
            &st,
            &pm,
            &sl,
            &nofvals_tmp,
            &src_saved,
            &dst_saved,
            &elapsed_total,
            &elapsed_scp,
            &last_task
    ) != 0) {
        fprintf(stderr, "Error: failed to load checkpoint from %s\n", INFO_PATH);
        return;
    }

    // printf("Checkpoint: %s\n", INFO_PATH);
    printf("Source: %s\n", src_saved ? src_saved : "-");
    printf("Destination: %s\n", dst_saved ? dst_saved : "-");
    printf("Inputs: %d, Outputs: %d\n", ni, no);
    // printf("Bits per word: %d, value bit width: %d, implicant words: %d\n", bpw, vbw, ipw);
        printf("Level k reached: %d (start level: %d, stop after max levels: %d)\n", ck, sl, ml);
    uint64_t maxt = nchoosek(ni, ck);

    if (ck > 0 && maxt > 0) {
        double pct = (double)(last_task + 1) * 100.0 / (double)maxt;
        if (pct > 100.0) pct = 100.0;
        printf("Progress at k: task=%llu / %llu (%.2f%%)\n", (unsigned long long)last_task, (unsigned long long)maxt, pct);
    }

    printf("Total time spent: %.3fs (%.3fs SCP)\n", elapsed_total, elapsed_scp);
    // printf("Stage: ready_for_coverage\n");

    if (info_level > 0) {
        for (int o = 0; o < no; ++o) {
                bool stop_flag = false;

            if (stopc_tmp) {
                stop_flag = stopc_tmp[o] >= ml;
            } else {
                stop_flag = pi_tmp[o].stop_search;
            }

            printf(
                "Output %d: ON=%d OFF=%d foundPI=%d solmin=%d stop=%s\n",
                o,
                pi_tmp[o].ON_minterms,
                pi_tmp[o].OFF_minterms,
                pi_tmp[o].foundPI,
                pi_tmp[o].solmin,
                stop_flag ? "yes" : "no"
            );
        }
    }

    if (src_saved) free(src_saved);
    if (dst_saved) free(dst_saved);
    if (nofvals_tmp) free(nofvals_tmp);
    if (stopc_tmp) free(stopc_tmp);

    // Use cleanup to free PInfo; provide a dummy buffer holder
    ThreadBuffer **dummy = (ThreadBuffer**)calloc(1, sizeof(ThreadBuffer*));
    cleanup(pi_tmp, dummy);
}


int process_task(
    uint64_t task,
    int k,
    int ninputs,
    int noutputs,
    int *nofvalues,
    int *bit_index,
    int *word_index,
    uint64_t *shifted_mask,
    int implicant_words,
    PIstorage *PInfo,
    ThreadBuffer **buffer,
    int tid,
    int *max_shared,
    int increase,
    int *multiplier
) {
    int tempk[k];
    uint64_t combination = task;

    DBG_TRACE_BLOCK {
        // if (task % 1000000 == 0 && task / 1000000 > 0) { // every 1M tasks
        //     fprintf(debug_out, "-");
        // }
        // if (task % 50000000 == 0 && task / 50000000 > 0) { // every 50M tasks
        //     fprintf(debug_out, "\n");
        // }
        // if (task % 100000000 == 0 && task / 100000000 > 0) { // every 100M tasks
        //     fprintf(debug_out, " (%lld)", task / 100000000);
        // }
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
        // fprintf(debug_out, "tempk: ");
        // for (int i = 0; i < k; i++) {
        //     fprintf(debug_out, "%d ", tempk[i] + 1);
        // }
        // fprintf(debug_out, "\n");
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
        int *decpos = (int *) calloc((size_t)ON_minterms, sizeof(int));
        int *decneg = (int *) calloc((size_t)OFF_minterms, sizeof(int));
        if (!decpos || !decneg) {
            fprintf(stderr, "Error: Memory allocation failed for decimal position arrays\n");
            return 1;
        }

        // create the vector of multiple bases, useful when calculating the decimal representation
        // of a particular combination of columns, for each row
        int mbase[k];
        mbase[0] = 1; // the first number is _always_ equal to 1, irrespective of the number of values in a certain input

        // calculate the vector of multiple bases, for example if we have k = 3 (three inputs) with
        // 2, 3 and 2 values then mbase will be [1, 2, 6] from: 1, 1 * 2 = 2, 2 * 3 = 6
        for (int i = 1; i < k; i++) {
            mbase[i] = mbase[i - 1] * nofvalues[tempk[i - 1]];
        }

        // Compute the total number of potential PIs for this combination: T = prod(v_s)
        int space_size = 1;
        for (int i = 0; i < k; i++) {
            space_size *= nofvalues[tempk[i]];
        }
        if (space_size < 1) space_size = 1;

        // Sum of mixed-radix bases (for normalizing decpos to 0..T-1 when values are 1..v)
        int mbase_sum = 0;
        for (int i = 0; i < k; i++) {
            mbase_sum += mbase[i];
        }

        // First pass: compute decpos for all ON rows (0 means invalid due to DC on selected inputs)
        // TODO: explore the potential of DC values compared to the OFF-set rows
        for (int r = 0; r < ON_minterms; r++) {
            int acc = 0;
            bool valid = true;

            for (int c = 0; c < k; c++) {
                int value = PInfo[o].ON_set[r * ninputs + tempk[c]];
                if (value == 0) {
                    valid = false;
                    break;
                }
                acc += value * mbase[c];
            }

            decpos[r] = valid ? acc : 0;
        }

        // calculate decimal numbers, using mbase, fills in decpos and decneg

        int unique_off_rows[OFF_minterms];
        bool dc_off_rows[OFF_minterms];
        int off_count = 0;

        // initialize don't-care flags to false
        for (int r = 0; r < OFF_minterms; r++) {
            dc_off_rows[r] = false;
        }

        // OFF-set O(1) dedup using the same mbase and space_size as ON-set
        bool *off_seen = (bool*)calloc((size_t)space_size, sizeof(bool));
        bool use_off_seen = (off_seen != NULL);

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

            size_t off_index = (size_t)acc; // normalized index in 0..space_size-1

            // O(1) uniqueness check using off_index; fallback to O(n) if allocation failed
            if (use_off_seen && off_index < (size_t)space_size) {
                if (off_seen[off_index]) {
                    continue; // duplicate OFF row pattern
                }

                off_seen[off_index] = true;
                unique_off_rows[off_count++] = r;
            } else {
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
        }

        if (off_seen) free(off_seen);


        int possible_rows[ON_minterms];
        int found = 0;

        // Use a visited set keyed by normalized decpos (0..space_size - 1) to skip duplicates
        bool *pos_seen = (bool*)calloc((size_t)space_size, sizeof(bool));
        // If allocation fails, we fallback to scanning duplicates (unlikely and still safe)
        bool use_seen = (pos_seen != NULL);

        for (int r = 0; r < ON_minterms; r++) {
            if (found >= space_size) break; // Early stop: all potential PIs already found
            if (decpos[r] == 0) continue;   // invalid row (has DC in selected inputs)

            int dec_norm = decpos[r] - mbase_sum;
            if (use_seen && dec_norm >= 0 && dec_norm < space_size && pos_seen[dec_norm]) {
                continue; // duplicate pattern already seen
            }

            if (!use_seen) {
                // O(n) fallback: check previously selected rows for duplicate decpos
                bool duplicate = false;

                for (int prev = 0; prev < found; prev++) {
                    if (decpos[possible_rows[prev]] == decpos[r]) {
                        duplicate = true;
                        break;
                    }
                }

                if (duplicate) continue;
            }

            // check if the row is different from any OFF-set row
            bool valid_row = true;
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

            possible_rows[found++] = r;
            max_found++;

            if (use_seen && dec_norm >= 0 && dec_norm < space_size) {
                pos_seen[dec_norm] = true;
            }

            if (found >= space_size) {
                break; // Guard also after increment
            }
        }

        if (pos_seen) free(pos_seen);

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

        free(decpos);
        free(decneg);
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
                // if (buffer[tid][o].found > 0) {
                //     fprintf(debug_out, "Output %d, found PIs:", o + 1);
                // }
            }

            for (int f = 0; f < buffer[tid][o].found; f++) {
                DBG_TRACE_BLOCK {
                    // fprintf(debug_out, " %d", buffer[tid][o].decpos[f]);
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
                // if (buffer[tid][o].found > 0) {
                //     fprintf(debug_out, "\n");
                // }
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

                        DBG_TRACE_BLOCK {
                            (*multiplier)++;
                            printf("%dx", *multiplier);
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
                    if (*max_shared < shared[*foundPI]) {
                        *max_shared = shared[*foundPI];
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

    return 0;
}