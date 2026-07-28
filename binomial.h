#ifndef BINOMIAL_H
#define BINOMIAL_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Prepare the read-only Pascal table used by worker-side combination
 * unranking. Call this before starting PI-generation workers.
 */
bool nchoosek_prepare(int max_n);

/*
 * Return C(n, k), or zero for invalid arguments and coefficients that do not
 * fit in uint64_t. Values up to the prepared limit are constant-time lookups.
 */
uint64_t nchoosek(int n, int k);

#endif
