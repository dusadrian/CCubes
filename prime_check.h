#ifndef PRIME_CHECK_H
#define PRIME_CHECK_H

#include <stdbool.h>
#include <stdint.h>

#include "utils.h"

/*
 * True when a candidate cube, fixed on the `level` inputs listed in
 * `support` with the binary literal values packed into `value_bits`, is a
 * prime implicant with respect to pi->OFF_set: no single literal can be
 * dropped without the cube starting to intersect an OFF-set row.
 */
bool projected_cube_is_prime(
    const PIstorage *pi,
    int ninputs,
    const int *support,
    int level,
    const uint64_t *value_bits,
    const int *word_index,
    const int *bit_index
);

#endif
