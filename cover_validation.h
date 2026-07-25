#ifndef COVER_VALIDATION_H
#define COVER_VALIDATION_H

#include <stdbool.h>

#include "pichart_view.h"

/*
 * Validate a set-cover incumbent against the bit-packed PI chart.
 * A valid cover contains distinct in-range column indices and covers every row.
 */
bool cover_is_feasible(
    const PIChartView *chart,
    const int solution[],
    int solution_size
);

#endif
