/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "utils.h"

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

    // Allocate ON_sets and OFF_sets
    *PInfo = malloc(outputs * sizeof(PIstorage));
    if (!*PInfo) {
        printf("Error: Memory allocation failed for PInfo\n");
        free(ON_minterms);
        free(OFF_minterms);
        fclose(file);
        return;
    }

    // Initialize all pointers to NULL for safe cleanup
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


