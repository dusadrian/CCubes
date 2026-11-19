#ifndef SYSTEMATIC_CERTIFY_H
#define SYSTEMATIC_CERTIFY_H

#include "utils.h"
#include "cover_encoding.h"

typedef struct {
    double shared_seconds; /* Zero uses the individual stage clocks. */
    double weighted_seconds;
    double sat_seconds;
    double verifier_seconds;
    uint64_t work;
    uint64_t weighted_work;
    uint64_t verifier_work;
    bool separate_verifier;
} SystematicBudget;

/* Certify the decoded cover in pi->solution. An improvement may replace
 * it with a smaller transposed cover. This operation is per scalar output. */
bool systematic_certify_outputs(PIstorage *pi, int ninputs, int noutputs,
                                int threads, const SystematicBudget *budget,
                                NsatCoverStrengthening mode,
                                bool reuse);

#ifdef CCUBES_SYSTEMATIC_MPI
typedef struct {
    int lower, terms;
    int *cover; /* Optional independently validated transposed cover. */
} SystematicSatResult;

SystematicSatResult systematic_sat_descent(const PIstorage *care, int inputs, int output,
                                          int incumbent, double deadline);
bool systematic_validate_cover(const PIstorage *pi, int inputs);
#endif

#endif
