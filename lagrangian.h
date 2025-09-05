#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

void solve_scp_lagrangian(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double col_costs[],
    const double weights[],
    int *solution,
    int *solmin
);
