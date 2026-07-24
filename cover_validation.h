#ifndef COVER_VALIDATION_H
#define COVER_VALIDATION_H

#include <stdbool.h>

/*
 * Validate a set-cover incumbent against the original column-major PI chart.
 * A valid cover contains distinct in-range column indices and covers every row.
 */
bool cover_is_feasible(
    const int pichart[],
    int cols,
    int rows,
    const int solution[],
    int solution_size
);

#endif
