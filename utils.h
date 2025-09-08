#ifndef UTILS_H
#define UTILS_H

#include <limits.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdalign.h> // For alignof

void error_message(const char *msg);

// Readable type identifiers
typedef enum {
    TYPE_BOOL,     // bool
    TYPE_INT,      // int
    TYPE_INT_ONES, // int initialized to 1
    TYPE_UINT64,   // uint64_t
    TYPE_DOUBLE    // double
} ArrayType;


typedef struct {
    // Basic info
    int inputs;
    int outputs;
    int ON_minterms;
    int OFF_minterms;

    int pichart_words;
    int estimPI;
    int foundPI;
    int solmin;
    int prevsolmin;
    bool stop_search;
    bool ON_set_covered;

    // Input data
    int      *ON_set;
    int      *OFF_set;
    int      *covered;
    int      *last_index;
    int      *k_last_index; // continued at each k
    int      *pichart;
    uint64_t *pichart_pos;
    uint64_t *implicants_pos;
    uint64_t *implicants_val;
    int      *shared;
    int      *covsum;
    int      *previndices;
    int      *indices;
    int      *cov_word_index;
    uint64_t *shifted_cov_mask;
    int      *nofpi;
    int      *k_covered;

    int *solution;
    int pool_count;
    int **pool_solutions;
} PIstorage;

typedef struct ThreadBuffer {
    int threads;
    uint64_t *pichart_values;
    bool     *coverage;
    int      *decpos;
    int      *covsum;
    uint64_t *fixed_bits;
    uint64_t *value_bits;
    int       found;
} ThreadBuffer;

uint64_t nchoosek(
    int n,
    int k
);

void resize(
    void **array,
    ArrayType type,
    int increase,
    int size,
    int nrows
);

void trim_whitespace(char *str);

void read_pla_file(
    const char *filename,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int **nofvalues,
    int *max_value
);

void write_pla_file(const char *filename, PIstorage *PInfo);

void cleanup(PIstorage *PInfo, ThreadBuffer **buffer);

char *prefix_basename(const char *filepath, const char *prefix);

void print_info(const char *path);

#endif // UTILS_H
