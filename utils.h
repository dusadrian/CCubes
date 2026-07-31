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
#include <stdatomic.h>

#include "debug.h"
#include "binomial.h"
#include "ccubes_threads.h"
#include "subsumption_index.h"
#include "pichart_view.h"

/*
 * Coverage bitsets (pichart_pos, and the per-thread pichart_values feeding
 * them) are stored in uint64_t words, so the number of coverage bits packed
 * per word can never exceed 64 -- shifting a uint64_t by 64 or more is
 * undefined behaviour, and on arm64 the shift count wraps modulo 64, which
 * silently aliases row r onto row r-64. The -b switch selects the *implicant*
 * packing width and is allowed to be 128, so coverage packing must be clamped
 * independently of it.
 */
static inline int coverage_bits_per_word(int bits_per_word) {
    return bits_per_word > 64 ? 64 : bits_per_word;
}

void error_message(const char *msg);

void destroy_output_locks(
    ccubes_mutex *locks,
    int noutputs
);

bool env_flag_enabled(
    const char *name
);

bool parse_int_strict(
    const char *text,
    int *value
);

bool parse_nonnegative_double(
    const char *text,
    double *value
);

bool parse_hybrid_effort_level(
    const char *text,
    int *level
);

void print_hybrid_stats(
    int output_index
);

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
    int cov_bits;       // coverage bits packed per pichart_pos word (<= 64)
    int estimPI;
    int foundPI;
    int solmin;
    int prevsolmin;
    bool stop_search;
    bool ON_set_covered;

    // Input data
    int      *ON_set;
    int      *OFF_set;
    /*
     * Exact input rows are shared across output-specific ON/OFF partitions.
     * projection_rows is owned only by PInfo[0]; every output owns its two
     * row-id maps into that common table.
     */
    int       projection_row_count;
    int      *projection_rows;
    int      *ON_projection_ids;
    int      *OFF_projection_ids;
    /*
     * DC-bearing OFF sets use compatibility masks indexed by
     * (input, non-DC value). A bit is set when the OFF row has that value or
     * a wildcard at the input. Fully specified outputs leave these fields
     * empty and retain the decoded-index validation path.
     */
    int       off_mask_words;
    int       off_mask_count;
    int      *off_mask_offsets;
    uint64_t *off_compat_masks;
    int      *covered;
    int      *last_index;
    int      *k_last_index; // continued at each k
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

/* Bit-packed chart view over the first `foundPI` columns of an output. */
static inline PIChartView pi_chart_view(const PIstorage *pi) {
    PIChartView view;
    view.bits = pi->pichart_pos;
    view.words = pi->pichart_words;
    view.cov_bits = pi->cov_bits;
    view.rows = pi->ON_minterms;
    view.cols = pi->foundPI;
    return view;
}

typedef struct ThreadBuffer {
    int threads;
    int capacity;
    uint64_t *pichart_values;
    int      *decpos;
    int      *covsum;
    uint64_t *fixed_bits;
    uint64_t *value_bits;
    int      *projection_codes;
    unsigned char *projection_has_dc;
    int       projection_capacity;
    int      *task_row_codes;
    size_t    task_row_capacity;
    uint32_t *task_seen_stamps;
    size_t    task_seen_capacity;
    uint32_t  task_seen_epoch;
    int       found;
#ifdef CCUBES_TESTING
    uint64_t  validation_attempts;
#endif
} ThreadBuffer;

/*
 * Per-output coverage index used while generating one complexity level.
 * Coverage keys are copied into the index so worker lookups remain valid if
 * the global PI arrays grow while a level is being generated.
 */
typedef struct {
    int *slots;
    uint64_t *keys;
    size_t table_size;
    size_t count;
    int words;
    SubsumptionIndex subsumption_index;
    atomic_uint_fast64_t subsumption_rejections;
} PICoverageIndex;

/* Binary total-row model required by adaptive and certified stopping. */
bool certified_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
);

/* Binary PLA patterns, allowing input dashes, for heuristic execution. */
bool heuristic_pattern_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
);

void resize(
    void **array,
    ArrayType type,
    int increase,
    int size,
    int nrows
);

double *build_cover_weights(
    const PIstorage *pi,
    int found_pi,
    int completed_level,
    int weight_mode
);

/*
 * Complete-prime enumeration is not grouped into contiguous complexity
 * levels.  Derive each secondary weight from the stored cube geometry instead
 * of nofpi[] level boundaries.
 */
double *build_complete_cover_weights(
    const PIstorage *pi,
    int found_pi,
    int ninputs,
    int implicant_words,
    const int *word_index,
    const uint64_t *shifted_mask,
    int weight_mode
);

/*
 * Pooling is a Boolean user choice.  Candidate discovery grows only
 * logarithmically with PI-chart width and is capped tightly because pooling
 * is a secondary multi-output objective, not part of cover feasibility.
 */
#define CCUBES_POOL_MIN_CANDIDATES 5
#define CCUBES_POOL_SEED_CANDIDATES 10
#define CCUBES_POOL_STORAGE_CAPACITY 20

int automatic_pool_solution_limit(
    int found_pi
);

void trim_whitespace(
    char *str
);

void read_pla_file(
    const char *filename,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int **nofvalues,
    int *max_value
);

bool prepare_shared_projection_rows(
    PIstorage *PInfo,
    int ninputs,
    int noutputs
);

bool prepare_off_wildcard_masks(
    PIstorage *PInfo,
    int ninputs,
    int noutputs,
    const int *nofvalues
);

void write_pla_file(
    const char *filename,
    PIstorage *PInfo
);

void cleanup(
    PIstorage *PInfo,
    ThreadBuffer **buffer
);

char *prefix_basename(
    const char *filepath,
    const char *prefix
);

void print_info(
    const char *path,
    const int info_level
);

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
    ccubes_mutex *output_locks,
    PICoverageIndex *coverage_indices,
    int *max_shared,
    int increase,
    int *multiplier
);

bool build_pi_coverage_indices(
    PICoverageIndex **indices,
    PIstorage *PInfo,
    int noutputs,
    const int *level_start,
    int implicant_words
);

void destroy_pi_coverage_indices(
    PICoverageIndex *indices,
    int noutputs
);

int finalize_pi_level(
    PIstorage *PInfo,
    int implicant_words,
    int level_start,
    bool deterministic_order,
    bool preserve_shared_geometries
);

int canonicalize_pi_order(
    PIstorage *PInfo,
    int implicant_words,
    int level_start,
    bool preserve_shared_geometries
);

#endif // UTILS_H
