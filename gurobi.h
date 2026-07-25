#include <assert.h>
#include <stdio.h>      // FILE, fopen, fclose, fflush
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>     // dup, dup2, close
#include <fcntl.h>

#include "pichart_view.h"
#ifdef HAVE_GUROBI
    #include "gurobi_c.h"
#endif

bool gurobi_license_is_valid(void);

void gurobi_multiobjective(
    const PIChartView *chart,
    double weights[],        // the weights for each individual PI
    const int *initial_solution, // optional validated incumbent column IDs
    int initial_solmin,
    int *indices,            // IDs of the selected prime implicants
    int *solmin              // no. of PIs covering the ON_minterms
);

void gurobi_solution_pool(
    const PIChartView *chart,
    const int max_pool,      // maximum number of solutions to collect
    double weights[],        // the weights for each individual PI
    const int *initial_solution, // optional validated incumbent column IDs
    int initial_solmin,
    int *pool_count,         // number of solutions returned (<= max_pool)
    int **pool_solutions,    // array of int* solutions
    int *solmin              // minimal number of PIs covering the ON_minterms
);
