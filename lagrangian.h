#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

void solve_scp_lagrangian(
    int pichart[],          // column-major (pichart[c * ON_minterms + r])
    const int foundPI,      // columns
    const int ON_minterms,  // rows
    const double weights[], // can be NULL; higher is better (tie-break inside heuristics)
    int *solution,          // out column indices (0-based)
    int *solmin             // out size (number of columns); -1 if infeasible
);

void solve_scp_lagrangian_pool(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double weights[],
    int max_pool,
    int *out_pool_count,
    int **pool_solutions,
    int *solmin
);
