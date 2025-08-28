/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "gurobi.h"

void gurobi_multiobjective(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    double weights[],        // the weights for each individual PI
    int *indices,            // IDs of the selected prime implicants
    int *solmin              // no. of PIs covering the ON_minterms
) {
    int error = 0;
    GRBenv      *env        = NULL;
    GRBmodel    *model      = NULL;
    int         *ind        = NULL;
    double      *coeffs     = NULL; // objective coefficients (reused)
    double      *gurobi_sol = NULL;

    ind  = malloc(foundPI * sizeof(int));
    coeffs  = malloc(foundPI * sizeof(double));
    gurobi_sol = (double*) malloc(foundPI * sizeof(double));

    if (!ind || !coeffs || !gurobi_sol) {
        error = 1;
        goto QUIT;
    }

    /* redirect stdout/stderr to suppress license banner */
    FILE *nullout = fopen("/dev/null", "w");
    int saved_stdout = dup(fileno(stdout));
    int saved_stderr = dup(fileno(stderr));
    dup2(fileno(nullout), fileno(stdout));
    dup2(fileno(nullout), fileno(stderr));

    error = GRBloadenv(&env, "/dev/null");
    if (error) goto QUIT;

    fflush(stdout); fflush(stderr);
    dup2(saved_stdout, fileno(stdout));
    dup2(saved_stderr, fileno(stderr));
    close(saved_stdout);
    close(saved_stderr);
    fclose(nullout);

    /* silence solver logs */
    error = GRBsetintparam(env, "OutputFlag", 0);
    if (error) goto QUIT;

    /* new model with foundPI variables */
    error = GRBnewmodel(
        env,
        &model,
        "SetCoveringProblem", // custom name
        foundPI,
        NULL, NULL, NULL, NULL, NULL
    );
    if (error) goto QUIT;

    /* binary vars */
    for (int j = 0; j < foundPI; j++) {
        error = GRBsetcharattrelement(model, GRB_CHAR_ATTR_VTYPE, j, GRB_BINARY);
        if (error) goto QUIT;
    }

    /* covering constraints: sum_{j | x[i,j]=1} x_j >= 1 for each row i */
    for (int i = 0; i < ON_minterms; i++) {
        int nz = 0;
        for (int j = 0; j < foundPI; j++) {
            if (pichart[i + ON_minterms * j] == 1) {
                ind[nz] = j;
                coeffs[nz] = 1.0;  // reuse coeffs[] as row coefficients
                nz++;
            }
        }
        error = GRBaddconstr(model, nz, ind, coeffs, GRB_GREATER_EQUAL, 1.0, NULL);
        if (error) goto QUIT;
    }

    /* first objective: minimize sum x_j */
    {
        for (int j = 0; j < foundPI; j++) {
            ind[j] = j;
            coeffs[j] = 1.0;
        }

        error = GRBsetobjectiven(
            model,
            0,          // objective index
            1,          // priority
            1.0,        // scaling factor (weight)
            0.0,        // abstol (absolute tolerance)
            0.0,        // reltol (relative tolerance)
            "mincols",  // name (optional, can be NULL)
            0.0,        // constant term, usually 0.0
            foundPI,    // number of nonzero coefficients
            ind,        // variable indices
            coeffs      // coefficient values
        );
    }

    /* second objective: maximize weighted value (if weights provided) */
    if (weights) {
        for (int j = 0; j < foundPI; j++) coeffs[j] = -1.0 * weights[j];

        error = GRBsetobjectiven(
            model,
            1,
            0,
            1.0,
            0.0,
            0.0,
            "maxweights",
            0.0, foundPI,
            ind,
            coeffs
        );
        if (error) goto QUIT;
    }

    // GRBwrite(model, "debug_c_model.lp");

    /* optimize */
    error = GRBoptimize(model);
    if (error) goto QUIT;

    double objval;
    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
    if (error) goto QUIT;

    // printf("Objective value: %.2f\n", objval);

    *solmin = (int)(objval);

    /* extract solution */
    error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, foundPI, gurobi_sol);
    if (error) goto QUIT;

    // printf("Optimal solution:\n");
    int pos = 0;
    for (int j = 0; j < foundPI; j++) {
        if (gurobi_sol[j] > 0.9) {
            indices[pos] = j;
            pos++;
            // printf("%d ", j+1);
        }
    }

    QUIT:

    if (ind) free(ind);
    if (coeffs) free(coeffs);
    if (gurobi_sol) free(gurobi_sol);
    if (error) {
        *solmin = 0;
        const char *errmsg = env ? GRBgeterrormsg(env) : "Unknown error (no env)";
        printf("ERROR: %s\n", errmsg);
    }
    if (model) GRBfreemodel(model);
    if (env) GRBfreeenv(env);
}


void gurobi_solution_pool(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const int max_solutions, // maximum number of solutions to find
    double weights[],        // the weights for each individual PI
    int *solcount,           // no. of solutions found
    int *solmat,             // solution matrix
    int *solmin              // no. of PIs covering the ON_minterms
) {
    int error = 0;
    GRBenv      *env        = NULL;
    GRBenv      *menv       = NULL; // model's env (for SolutionNumber)
    GRBmodel    *model      = NULL;
    int         *ind        = NULL;
    double      *coeffs     = NULL; // objective coefficients (reused)
    double      *gurobi_sol = NULL;

    ind  = malloc(foundPI * sizeof(int));
    coeffs  = malloc(foundPI * sizeof(double));
    gurobi_sol = (double*) malloc(foundPI * sizeof(double));

    if (!ind || !coeffs || !gurobi_sol) {
        error = 1;
        goto QUIT;
    }

    // for (int r = 0; r < ON_minterms; r++) {
    //     for (int c = 0; c < foundPI; c++) {
    //         printf("%d ", pichart[r + ON_minterms * c]);
    //     }
    //     printf("\n");
    // }

    /* Try to initialize Gurobi environment with better error handling */
    error = GRBloadenv(&env, NULL);  // Use NULL instead of "/dev/null" to avoid file issues
    if (error) {
        printf("ERROR: Failed to create Gurobi environment: %s\n", GRBgeterrormsg(env));
        goto QUIT;
    }

    /* silence solver logs */
    error = GRBsetintparam(env, "OutputFlag", 0);
    if (error) goto QUIT;

    /* new model with foundPI variables */
    error = GRBnewmodel(
        env,
        &model,
        "SetCoveringProblem", // custom name
        foundPI,
        NULL, NULL, NULL, NULL, NULL
    );
    if (error) goto QUIT;

    menv = GRBgetenv(model);
    assert(menv != NULL);


    /* binary vars */
    for (int j = 0; j < foundPI; j++) {
        error = GRBsetcharattrelement(model, GRB_CHAR_ATTR_VTYPE, j, GRB_BINARY);
        if (error) goto QUIT;
    }

    /* covering constraints: sum_{j | x[i,j]=1} x_j >= 1 for each row i */
    for (int i = 0; i < ON_minterms; i++) {
        int nz = 0;
        for (int j = 0; j < foundPI; j++) {
            if (pichart[i + ON_minterms * j] == 1) {
                ind[nz] = j;
                coeffs[nz] = 1.0;  // reuse coeffs[] as row coefficients
                nz++;
            }
        }
        error = GRBaddconstr(model, nz, ind, coeffs, GRB_GREATER_EQUAL, 1.0, NULL);
        if (error) goto QUIT;
    }

    /* ---------- Phase 1: single objective MIN sum x_j ---------- */
    for (int j = 0; j < foundPI; j++) coeffs[j] = 1.0;

    // set linear objective coefficients
    error = GRBsetdblattrarray(model, GRB_DBL_ATTR_OBJ, 0, foundPI, coeffs);
    if (error) goto QUIT;

    // set sense to minimize
    error = GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE);
    if (error) goto QUIT;

    error = GRBupdatemodel(model); // ensure objective/constraints are in
    if (error) goto QUIT;

    error = GRBoptimize(model);
    if (error) goto QUIT;

    double objval;
    error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
    if (error) goto QUIT;

    *solmin = (int)(objval);
    // printf("Solution minima: %.0f\n", *solmin);

    /* add cardinality constraint: sum x_j <= solmin */
    {
        for (int j = 0; j < foundPI; j++) ind[j] = j;
        for (int j = 0; j < foundPI; j++) coeffs[j] = 1.0;

        error = GRBaddconstr(model, foundPI, ind, coeffs, GRB_LESS_EQUAL, *solmin, NULL);
        if (error) goto QUIT;
    }

    /* ---------- Phase 2: single objective MAX value^T x, with solution pool ---------- */

    // set the *single* objective to maximize weighted value
    error = GRBsetdblattrarray(model, GRB_DBL_ATTR_OBJ, 0, foundPI, weights);
    if (error) goto QUIT;
    error = GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE);
    if (error) goto QUIT;

    // enable solution pool (set on the model's env)
    error = GRBsetintparam(menv, GRB_INT_PAR_POOLSOLUTIONS, max_solutions);
    if (error) goto QUIT;
    error = GRBsetintparam(menv, GRB_INT_PAR_POOLSEARCHMODE, 2);
    if (error) goto QUIT;

    error = GRBupdatemodel(model);
    if (error) goto QUIT;

    error = GRBoptimize(model);
    if (error) goto QUIT;


    /* get number of solutions in pool */
    error = GRBgetintattr(model, GRB_INT_ATTR_SOLCOUNT, solcount);
    if (error) goto QUIT;

    // printf("Found %d solutions in pool (max 100)\n", solcount);

    /* loop over solutions in pool */
    for (int s = 0; s < *solcount; s++) {

        // Use model's environment when selecting solution from pool
        error = GRBsetintparam(menv, GRB_INT_PAR_SOLUTIONNUMBER, s);
        if (error) goto QUIT;

        // Query number of variables and size buffer accordingly
        int numvars = 0;
        error = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &numvars);
        if (error) goto QUIT;
        if (numvars < 0) { error = 1; goto QUIT; }
        if (numvars > foundPI) {
            double *tmp = (double*)realloc(gurobi_sol, (size_t)numvars * sizeof(double));
            if (!tmp) { error = 1; goto QUIT; }
            gurobi_sol = tmp;
        }
        int len = (numvars < foundPI) ? numvars : foundPI;

        error = GRBgetdblattrarray(model, GRB_DBL_ATTR_XN, 0, len, gurobi_sol);
        if (error) goto QUIT;

        // printf("Solution %d:\n", s + 1);
        int pos = 0;
        for (int j = 0; j < len; j++) {
            if (gurobi_sol[j] > 0.99) {
                if (pos < ON_minterms) {
                    solmat[s * ON_minterms + pos] = j;
                    // printf(" %d", j + 1);
                    pos++;
                } else {
                    break; // prevent overflow
                }
            }
        }

        // printf("\n");
    }

    QUIT:
    if (ind) free(ind);
    if (coeffs) free(coeffs);
    if (gurobi_sol) free(gurobi_sol);
    if (error) {
        printf("ERROR: %s\n", GRBgeterrormsg(env));
    }
    if (model) GRBfreemodel(model);
    if (env) GRBfreeenv(env);
}
