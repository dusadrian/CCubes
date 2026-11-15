#include "cover_encoding.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NsatLimits unlimited = {0};

#ifdef NSAT_TESTING
/* Only the implementation's accounted allocations use this entry point. */
static long allocations_before_failure = -1;

void *nsat_test_malloc(size_t bytes)
{
    if (allocations_before_failure == 0) return NULL;
    if (allocations_before_failure > 0) --allocations_before_failure;
    return malloc(bytes);
}
#endif

static void add_pair(NsatSolver *s, NsatVerifier *v, const int *lits, size_t count)
{
    assert(nsat_add_original(s, lits, count) == NSAT_OK);
    assert(nsat_verify_add_original(v, lits, count) == NSAT_OK);
}

typedef struct {
    const int *lits;
    size_t count;
} ClauseView;

/* Independent full-scan oracle. Values are indexed by variable, with
 * 0 for unassigned and +/-1 for true/false. */
static bool full_scan(const ClauseView *clauses, size_t count, int *values)
{
    bool changed;
    do {
        changed = false;
        for (size_t c = 0; c < count; ++c) {
            int unit = 0, open = 0;
            bool satisfied = false;
            for (size_t i = 0; i < clauses[c].count; ++i) {
                int l = clauses[c].lits[i];
                int value = l > 0 ? values[l] : -values[-l];
                if (value > 0) { satisfied = true; break; }
                if (!value) { ++open; unit = l; }
            }
            if (satisfied) continue;
            if (!open) return false;
            if (open == 1) {
                values[abs(unit)] = unit < 0 ? -1 : 1;
                changed = true;
            }
        }
    } while (changed);
    return true;
}

enum { TEST_VARS = 177 };

static bool reference_sat(const ClauseView *clauses, size_t count, int *values);
static unsigned root_cases, search_cases;

static void check_model(const NsatSolver *s, int variables,
                        const ClauseView *clauses, size_t count)
{
    for (int i = 1; i <= variables; ++i)
        assert(abs(nsat_model_value(s, i)) == 1);
    for (size_t c = 0; c < count; ++c) {
        bool satisfied = false;
        for (size_t i = 0; i < clauses[c].count; ++i) {
            int lit = clauses[c].lits[i];
            if (nsat_model_value(s, abs(lit)) == (lit < 0 ? -1 : 1))
                satisfied = true;
        }
        assert(satisfied);
    }
}

static void compare_search(const ClauseView *clauses, size_t count, int variables,
                           NsatSearchOptions options)
{
    int values[TEST_VARS] = {0};
    bool satisfiable = reference_sat(clauses, count, values);
    NsatSolver *s;
    NsatVerifier *v;
    assert(nsat_init(&s, variables, unlimited) == NSAT_OK);
    assert(nsat_verify_init_cnf(&v, variables, unlimited) == NSAT_OK);
    for (size_t i = 0; i < count; ++i)
        add_pair(s, v, clauses[i].lits, clauses[i].count);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    NsatStatus result = nsat_solve(s, v, options);
    if (result != (satisfiable ? NSAT_SAT : NSAT_UNSAT)) {
        fprintf(stderr, "search case %u: status=%d stop=%d expected_satisfiable=%d\n",
                search_cases, result, nsat_stats(s).stop, satisfiable);
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < clauses[i].count; ++j)
                fprintf(stderr, "%d ", clauses[i].lits[j]);
            fputs("0\n", stderr);
        }
        abort();
    }
    if (result == NSAT_SAT) check_model(s, variables, clauses, count);
    else assert(nsat_model_value(s, 1) == 0);
    nsat_destroy(s);
    nsat_verify_destroy(v);
    ++search_cases;
}

static void compare_root(const ClauseView *clauses, size_t count, int variables)
{
    NsatSolver *s;
    NsatVerifier *v;
    assert(nsat_init(&s, variables, unlimited) == NSAT_OK);
    assert(nsat_verify_init_cnf(&v, variables, unlimited) == NSAT_OK);
    for (size_t i = 0; i < count; ++i)
        add_pair(s, v, clauses[i].lits, clauses[i].count);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    int values[4] = {0};
    bool consistent = full_scan(clauses, count, values);
    NsatStatus result = nsat_propagate_root(s, v);
    assert(result == (consistent ? NSAT_ROOT_FIXED_POINT : NSAT_UNSAT));
    if (consistent) {
        for (int i = 1; i <= variables; ++i) {
            assert(nsat_root_value(s, i) == values[i]);
            NsatClauseId id = nsat_root_unit_id(s, i);
            if (values[i]) {
                const int *lits;
                size_t width;
                assert(nsat_verify_clause(v, id, &lits, &width));
                assert(width == 1 && lits[0] == i * values[i]);
            } else assert(!id);
        }
    }
    nsat_destroy(s);
    nsat_verify_destroy(v);
    ++root_cases;
}

static void enumerate_sets(const ClauseView *pool, size_t size, size_t start,
                           ClauseView *selection, size_t length, size_t limit,
                           int variables)
{
    compare_root(selection, length, variables);
    compare_search(selection, length, variables, (NsatSearchOptions){0});
    if (length == limit) return;
    for (size_t i = start; i < size; ++i) {
        selection[length] = pool[i];
        enumerate_sets(pool, size, i + 1, selection, length + 1, limit, variables);
    }
}

static void test_root(void)
{
    for (int n = 2; n <= 3; ++n) {
        int literals[27][3];
        ClauseView pool[27], selected[9];
        int clauses = n == 2 ? 9 : 27;
        for (int c = 0; c < clauses; ++c) {
            size_t count = 0;
            for (int i = 1, code = c; i <= n; ++i, code /= 3) {
                if (code % 3) literals[c][count++] = code % 3 == 1 ? i : -i;
            }
            pool[c] = (ClauseView){literals[c], count};
        }
        enumerate_sets(pool, (size_t)clauses, 0, selected, 0, n == 2 ? 9 : 4, n);
    }
    assert(root_cases == 512 + 20854);

    /* No original ID changes when a width-three clause becomes a unit. */
    const int a[] = {-1}, b[] = {-2}, c[] = {1, 2, 3};
    compare_root((ClauseView[]){{a, 1}, {c, 3}, {b, 1}}, 3, 3);
    /* The false watch is relocated before a later implication falsifies it. */
    const int imply[] = {1, -3};
    compare_root((ClauseView[]){{c, 3}, {a, 1}, {imply, 2}}, 3, 3);
}

static bool reference_sat(const ClauseView *clauses, size_t count, int *values)
{
    if (!full_scan(clauses, count, values)) return false;
    int choice = 0;
    for (size_t c = 0; !choice && c < count; ++c) {
        int open = 0;
        bool satisfied = false;
        for (size_t i = 0; i < clauses[c].count; ++i) {
            int lit = clauses[c].lits[i];
            if (values[abs(lit)] == (lit < 0 ? -1 : 1)) {
                satisfied = true;
                break;
            }
            if (!values[abs(lit)]) open = lit;
        }
        if (!satisfied) choice = open;
    }
    if (!choice) return true;
    int saved[TEST_VARS];
    memcpy(saved, values, sizeof(saved));
    values[abs(choice)] = choice < 0 ? -1 : 1;
    if (reference_sat(clauses, count, values)) return true;
    memcpy(values, saved, sizeof(saved));
    values[abs(choice)] = choice < 0 ? 1 : -1;
    return reference_sat(clauses, count, values);
}

/* Enumerate ternary cubes directly and find the fewest reaching each
 * subset of positive rows. This oracle never uses SAT variables. */
static int reference_cover(int n, unsigned on, unsigned off)
{
    int best[256];
    for (int i = 0; i < 256; ++i) best[i] = 9;
    best[0] = 0;
    int cubes = n == 1 ? 3 : n == 2 ? 9 : 27;
    unsigned legal[27];
    size_t count = 0;
    for (int cube = 0; cube < cubes; ++cube) {
        unsigned covered = 0;
        for (int row = 0; row < (1 << n); ++row) {
            int code = cube, i = 0;
            for (; i < n; ++i, code /= 3) {
                int digit = code % 3;
                if (digit && digit != 1 + ((row >> i) & 1)) break;
            }
            if (i == n) covered |= 1u << row;
        }
        if (!(covered & off) && (covered & on)) legal[count++] = covered & on;
    }
    for (unsigned have = 0; have < on; ++have) {
        if (best[have] >= 8) continue;
        for (size_t i = 0; i < count; ++i) {
            unsigned more = have | legal[i];
            if (best[more] > best[have] + 1) best[more] = best[have] + 1;
        }
    }
    return best[on];
}

static bool model_covers(const NsatSolver *s, const int *point, int n, int k)
{
    for (int t = 0; t < k; ++t) {
        bool covers = true;
        for (int i = 0; i < n; ++i) {
            int base = 2 * (n * t + i);
            if (nsat_model_value(s, base + 3 - point[i]) > 0) {
                covers = false;
                break;
            }
        }
        if (covers) return true;
    }
    return false;
}

static unsigned incremental_queries;

static void compare_descent(const int *on, size_t p, const int *off, size_t q,
                            int n, int optimum)
{
    size_t start = (size_t)optimum + 2;
    if (start > p) start = p;
    NsatCoverInput input = {.on = on, .off = off, .inputs = (size_t)n,
        .positives = p, .negatives = q, .target = start};
    NsatSolver *solver;
    NsatVerifier *checker;
    NsatStopReason stop;
    NsatCoverDomainStats domain;
    NsatStatus result = nsat_cover_solve(&input, NSAT_COVER_COMBINED, unlimited,
                                        &solver, &checker, &stop, &domain);
    int variables = nsat_verify_variables(checker);
    NsatClauseId originals = nsat_verify_original_count(checker);
    for (size_t target = start;; --target) {
        ++incremental_queries;
        assert(nsat_verify_cover_slots(checker) == start);
        assert(nsat_verify_cover_target(checker) == target);
        assert(nsat_verify_original_count(checker) == originals);
        assert(nsat_verify_variables(checker) == variables);
        assert(result == (target >= (size_t)optimum ? NSAT_SAT : NSAT_UNSAT));
        if (result == NSAT_UNSAT) {
            assert(nsat_model_value(solver, 1) == 0);
            break;
        }
        for (size_t i = 0; i < p; ++i) assert(model_covers(solver, on + i * n, n, (int)target));
        for (size_t i = 0; i < q; ++i) assert(!model_covers(solver, off + i * n, n, (int)target));
        if (target == 1) break;
        result = nsat_solve_tighter_cover(solver, checker, target - 1);
    }
    nsat_destroy(solver);
    nsat_verify_destroy(checker);
}

static void test_cover(void)
{
    unsigned tables = 0, queries = 0;
    for (int n = 1; n <= 3; ++n) {
        const int classifications = n == 1 ? 9 : n == 2 ? 81 : 6561;
        for (int table = 0; table < classifications; ++table) {
            int on[24], off[24];
            size_t p = 0, q = 0;
            unsigned positives = 0, negatives = 0;
            for (int row = 0, code = table; row < (1 << n); ++row, code /= 3) {
                int digit = code % 3;
                if (!digit) continue;
                int *point = digit == 1 ? on + p++ * n : off + q++ * n;
                for (int i = 0; i < n; ++i) point[i] = 1 + ((row >> i) & 1);
                if (digit == 1) positives |= 1u << row;
                else negatives |= 1u << row;
            }
            int optimum = reference_cover(n, positives, negatives);
            ++tables;
            if (!p || !q) {
                assert(optimum == (p != 0));
                continue;
            }
            compare_descent(on, p, off, q, n, optimum);
            size_t order[8];
            for (size_t i = 0; i < p; ++i) order[i] = p - 1 - i;
            for (int k = 1; k <= optimum; ++k) {
                NsatVerifier *v;
                assert(nsat_verify_init_cover(&v, on, off, (size_t)n, p, q,
                                              (size_t)k, order, p, unlimited) == NSAT_OK);
                size_t count = (size_t)nsat_verify_original_count(v);
                ClauseView *clauses = malloc(count * sizeof(*clauses));
                assert(clauses && nsat_verify_variables(v) < TEST_VARS);
                for (size_t c = 0; c < count; ++c)
                    assert(nsat_verify_clause(v, c + 1, &clauses[c].lits, &clauses[c].count));
                int values[TEST_VARS] = {0};
                assert(reference_sat(clauses, count, values) == (k == optimum));
                NsatSolver *solver;
                NsatCoverInput input = {.on = on, .off = off, .inputs = (size_t)n,
                    .positives = p, .negatives = q, .target = (size_t)k,
                    .row_order = order, .order_count = p};
                NsatStopReason stop;
                assert(nsat_cover_load_solver(&solver, &input, v, unlimited, &stop) == NSAT_OK);
                NsatStatus result = nsat_solve(solver, v, (NsatSearchOptions){0});
                if (result != (k == optimum ? NSAT_SAT : NSAT_UNSAT)) {
                    fprintf(stderr, "cover n=%d table=%d K=%d: status=%d stop=%d\n",
                            n, table, k, result, nsat_stats(solver).stop);
                    abort();
                }
                if (result == NSAT_SAT) {
                    check_model(solver, nsat_verify_variables(v), clauses, count);
                    for (size_t i = 0; i < p; ++i)
                        assert(model_covers(solver, on + i * n, n, k));
                    for (size_t i = 0; i < q; ++i)
                        assert(!model_covers(solver, off + i * n, n, k));
                }
                nsat_destroy(solver);
                free(clauses);
                nsat_verify_destroy(v);

                input.row_order = NULL;
                input.order_count = 0;
                NsatCoverDomainStats domain;
                result = nsat_cover_solve(&input, NSAT_COVER_COMBINED, unlimited,
                                          &solver, &v, &stop, &domain);
                assert(result == (k == optimum ? NSAT_SAT : NSAT_UNSAT));
                if (result == NSAT_SAT) {
                    for (size_t i = 0; i < p; ++i)
                        assert(model_covers(solver, on + i * n, n, k));
                    for (size_t i = 0; i < q; ++i)
                        assert(!model_covers(solver, off + i * n, n, k));
                }
                nsat_destroy(solver);
                nsat_verify_destroy(v);
                ++queries;
            }
        }
    }
    printf("Reference oracles: %u root formulas, %u CNF searches, "
           "%u care tables, %u cover queries, %u incremental stages\n",
           root_cases, search_cases, tables, queries, incremental_queries);
}

enum { GATE_INPUTS = 20, GATE_OUTPUTS = 10, GATE_CARE = 40, GATE_TERMS = 41 };

typedef struct {
    char inputs[GATE_INPUTS + 1];
    char outputs[GATE_OUTPUTS + 1];
} GateRow;

static void read_gate_rows(const char *path, GateRow *rows, size_t expected)
{
    FILE *input = fopen(path, "r");
    assert(input);
    char line[128];
    size_t used = 0;
    while (fgets(line, sizeof(line), input)) {
        if (line[0] == '.') continue;
        assert(used < expected);
        assert(sscanf(line, "%20[01-] %10[01]", rows[used].inputs, rows[used].outputs) == 2);
        assert(strlen(rows[used].inputs) == GATE_INPUTS);
        assert(strlen(rows[used].outputs) == GATE_OUTPUTS);
        ++used;
    }
    assert(!ferror(input) && used == expected);
    fclose(input);
}

static bool gate_cube_covers(const GateRow *cube, const GateRow *point)
{
    for (size_t i = 0; i < GATE_INPUTS; ++i) {
        if (cube->inputs[i] != '-' && cube->inputs[i] != point->inputs[i]) return false;
    }
    return true;
}

static void test_native20_gate(void)
{
    GateRow care[GATE_CARE], terms[GATE_TERMS];
    read_gate_rows("examples/rnd_20x10x40.pla", care, GATE_CARE);
    read_gate_rows("tests/fixtures/native20-incumbent.pla", terms, GATE_TERMS);
    const size_t incumbents[GATE_OUTPUTS] = {5, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    uint64_t work[2] = {0}, deleted[2] = {0}, moved[2] = {0}, restarts[2] = {0};
    for (size_t o = 0; o < GATE_OUTPUTS; ++o) {
        size_t cubes = 0;
        for (size_t t = 0; t < GATE_TERMS; ++t) cubes += terms[t].outputs[o] == '1';
        assert(cubes == incumbents[o]);
        size_t p = 0, q = 0;
        int on[GATE_CARE * GATE_INPUTS], off[GATE_CARE * GATE_INPUTS];
        for (size_t r = 0; r < GATE_CARE; ++r) {
            bool positive = care[r].outputs[o] == '1';
            bool covered = false;
            for (size_t t = 0; t < GATE_TERMS; ++t) {
                if (terms[t].outputs[o] == '1' && gate_cube_covers(&terms[t], &care[r]))
                    covered = true;
            }
            assert(positive == covered);
            int *point = positive ? on + GATE_INPUTS * p++ : off + GATE_INPUTS * q++;
            for (size_t i = 0; i < GATE_INPUTS; ++i) {
                assert(care[r].inputs[i] == '0' || care[r].inputs[i] == '1');
                point[i] = 1 + care[r].inputs[i] - '0';
            }
        }
        NsatCoverInput input = {.on = on, .off = off, .inputs = GATE_INPUTS,
            .positives = p, .negatives = q, .target = cubes - 1};
        for (size_t stress = 0; stress <= 1; ++stress) {
            NsatResources resources = {.work_limit = 100000000, .memory_limit = 128u << 20};
            NsatLimits limits = {.shared = &resources};
            NsatSearchOptions options = {.restart_base = stress, .reduction_interval = stress};
            NsatVerifier *v;
            NsatSolver *s;
            NsatStopReason stop;
            assert(nsat_verify_init_cover(&v, on, off, input.inputs, p, q,
                                          input.target, NULL, 0, limits) == NSAT_OK);
            assert(nsat_cover_load_solver(&s, &input, v, limits, &stop) == NSAT_OK);
            NsatStatus status = nsat_solve(s, v, options);
            if (status != NSAT_UNSAT) {
                fprintf(stderr, "20-input output=%zu stress=%zu status=%d stop=%d work=%llu\n",
                        o + 1, stress, status, resources.stop, (unsigned long long)resources.work);
                abort();
            }
            NsatSearchStats stats = nsat_search_stats(s);
            assert(nsat_verify_stats(v).clauses_deleted == stats.deleted);
            assert(!stats.deleted || (stats.compactions && stats.words_reclaimed));
            work[stress] += resources.work;
            deleted[stress] += stats.deleted;
            moved[stress] += stats.moved;
            restarts[stress] += stats.restarts;
            assert(resources.peak_memory <= resources.memory_limit);
            nsat_destroy(s);
            nsat_verify_destroy(v);
            assert(resources.memory == 0);
        }
    }
    assert(deleted[1] && moved[1] && restarts[1]);
    printf("Native-only 20-input gate: 10/10 checked in both profiles "
           "(work %llu/%llu, deleted %llu/%llu, relocated %llu/%llu, restarts %llu/%llu)\n",
           (unsigned long long)work[0], (unsigned long long)work[1],
           (unsigned long long)deleted[0], (unsigned long long)deleted[1],
           (unsigned long long)moved[0], (unsigned long long)moved[1],
           (unsigned long long)restarts[0], (unsigned long long)restarts[1]);
}

static void compare_domain(const int *on, size_t p, const int *off, size_t q,
                           int n, int k, NsatStatus expected,
                           bool require_pair, bool require_triple)
{
    NsatCoverInput input = {.on = on, .off = off, .inputs = (size_t)n,
        .positives = p, .negatives = q, .target = (size_t)k};
    for (int mode = NSAT_COVER_GROUPS; mode <= NSAT_COVER_COMBINED; ++mode) {
        NsatResources resources = {0};
        NsatLimits limits = {.shared = &resources};
        NsatSolver *s;
        NsatVerifier *v;
        NsatStopReason stop;
        NsatCoverDomainStats stats;
        NsatStatus status = nsat_cover_solve(&input, (NsatCoverStrengthening)mode,
                                           limits, &s, &v, &stop, &stats);
        assert(status == expected);
        if (mode != NSAT_COVER_ORDERING) {
            if (require_pair) assert(stats.pairs > 0);
            if (require_triple) assert(stats.triples > 0 && stats.pairs == 0);
            if (require_pair && expected == NSAT_UNSAT)
                assert(nsat_search_stats(s).conflicts == 0);
        }
        if (mode != NSAT_COVER_GROUPS) {
            assert(stats.anchors > 0);
            assert(stats.staircase == 2 * (p - 1) * (size_t)(k - 1));
        }
        if (status == NSAT_SAT) {
            for (size_t i = 0; i < p; ++i) assert(model_covers(s, on + i * n, n, k));
            for (size_t i = 0; i < q; ++i) assert(!model_covers(s, off + i * n, n, k));
        }
        nsat_destroy(s);
        nsat_verify_destroy(v);
        assert(resources.memory == 0);
    }
}

static void test_domain_lemmas(void)
{
    const int triple_on[] = {2,2,1, 2,1,2, 1,2,2}, zero[] = {1,1,1};
    compare_domain(triple_on, 3, zero, 1, 3, 1, NSAT_UNSAT, false, true);
    compare_domain(triple_on, 3, zero, 1, 3, 2, NSAT_SAT, false, true);
    const int clique_on[] = {2,1,1, 1,2,1, 1,1,2};
    compare_domain(clique_on, 3, zero, 1, 3, 2, NSAT_UNSAT, true, false);
    compare_domain(clique_on, 3, zero, 1, 3, 3, NSAT_SAT, true, false);

    enum { width = 513 };
    const size_t active[] = {0, 64, width - 1};
    int wide_triple[3 * width], wide_clique[3 * width], wide_off[width];
    for (size_t i = 0; i < 3 * width; ++i) wide_triple[i] = wide_clique[i] = 1;
    for (size_t i = 0; i < width; ++i) wide_off[i] = 1;
    for (size_t r = 0; r < 3; ++r) {
        for (size_t i = 0; i < 3; ++i) {
            wide_triple[r * width + active[i]] = triple_on[r * 3 + i];
            wide_clique[r * width + active[i]] = clique_on[r * 3 + i];
        }
    }
    compare_domain(wide_triple, 3, wide_off, 1, width, 1, NSAT_UNSAT, false, true);
    compare_domain(wide_triple, 3, wide_off, 1, width, 2, NSAT_SAT, false, true);
    compare_domain(wide_clique, 3, wide_off, 1, width, 2, NSAT_UNSAT, true, false);

    /* A trusted checker may be bound only once. A failed root lemma
     * cannot enter the search or produce a conclusion. */
    NsatSolver *s;
    NsatVerifier *v;
    assert(nsat_init(&s, 1, unlimited) == NSAT_OK);
    assert(nsat_verify_init_cnf(&v, 1, unlimited) == NSAT_OK);
    add_pair(s, v, (int[]){1}, 1);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    NsatClauseId id = 99;
    assert(nsat_add_root_rup(s, v, (int[]){-1}, 1, (NsatClauseId[]){1}, 1, &id) == NSAT_ERROR);
    assert(id == 0 && nsat_solve(s, v, (NsatSearchOptions){0}) == NSAT_ERROR);
    nsat_destroy(s);
    nsat_verify_destroy(v);
}

static void test_cover_validation(void)
{
    NsatVerifier *v;
    int on[] = {1, 1, 2, 2};
    const int off[] = {1, 2};
    const size_t bad[][2] = {{0, 0}, {0, 2}};
    for (size_t i = 0; i < 2; ++i) {
        assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, bad[i], 2, unlimited)
               == NSAT_ERROR);
        nsat_verify_destroy(v);
    }
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, bad[0], 1, unlimited)
           == NSAT_ERROR && !v);
    assert(nsat_verify_init_cover(&v, on, off, SIZE_MAX, 2, 1, 1, NULL, 0, unlimited)
           == NSAT_ERROR && !v);
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited)
           == NSAT_OK);
    const int *lits;
    size_t count;
    assert(nsat_verify_original_count(v) == 14);
    assert(nsat_verify_clause(v, 14, &lits, &count) && count == 2);
    assert(lits[0] == 2 && lits[1] == 3);
    assert(nsat_verify_add_original(v, NULL, 0) == NSAT_ERROR);
    nsat_verify_destroy(v);
    on[1] = 2;
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited)
           == NSAT_ERROR);
    nsat_verify_destroy(v);
    on[1] = 0;
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited)
           == NSAT_ERROR);
    nsat_verify_destroy(v);
}

static void load_two_cube_control(NsatSolver **s, NsatVerifier **v, NsatLimits limits)
{
    const int on[] = {1,1, 2,2}, off[] = {1,2};
    NsatCoverInput input = {.on = on, .off = off, .inputs = 2,
        .positives = 2, .negatives = 1, .target = 2};
    NsatStopReason stop;
    NsatCoverDomainStats stats;
    assert(nsat_cover_solve(&input, NSAT_COVER_BASELINE, limits, s, v, &stop, &stats) == NSAT_SAT);
}

static void test_incremental_lifecycle(void)
{
    NsatSolver *s;
    NsatVerifier *v;
    NsatClauseId id;
    int unit;
    load_two_cube_control(&s, &v, unlimited);
    assert(nsat_solve_tighter_cover(s, v, 1) == NSAT_UNSAT);
    assert(nsat_solve_tighter_cover(s, v, 1) == NSAT_ERROR);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    /* An old SAT assignment satisfies the wide formula but violates the
     * tighter query. The independent checker must reject that model. */
    load_two_cube_control(&s, &v, unlimited);
    int vars = nsat_verify_variables(v);
    assert(vars == 16);
    int model[16];
    for (int i = 1; i <= vars; ++i) model[i - 1] = i * nsat_model_value(s, i);
    assert(nsat_verify_tighten_cover(v, 1, &id, &unit) == NSAT_OK);
    assert(unit == -vars && model[vars - 1] == vars);
    assert(nsat_verify_conclude_sat(v, model, 16) == NSAT_ERROR);
    assert(nsat_verify_stats(v).stop == NSAT_STOP_PROOF);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    load_two_cube_control(&s, &v, unlimited);
    assert(nsat_verify_tighten_cover(v, 1, &id, &unit) == NSAT_OK);
    assert(nsat_verify_delete(v, id) == NSAT_ERROR);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    for (size_t invalid = 0; invalid <= 3; ++invalid) {
        if (invalid == 1) continue;
        load_two_cube_control(&s, &v, unlimited);
        assert(nsat_solve_tighter_cover(s, v, invalid) == NSAT_ERROR);
        assert(nsat_model_value(s, 1) == 0);
        nsat_destroy(s);
        nsat_verify_destroy(v);
    }

    NsatResources resources = {0};
    load_two_cube_control(&s, &v, (NsatLimits){.shared = &resources});
    resources.work_limit = resources.work;
    assert(nsat_solve_tighter_cover(s, v, 1) == NSAT_UNKNOWN);
    assert(nsat_stats(s).stop == NSAT_STOP_WORK && nsat_model_value(s, 1) == 0);
    nsat_destroy(s);
    nsat_verify_destroy(v);
    assert(!resources.memory);
}

static void test_encoder_rejects_different_originals(void)
{
    const int on[] = {1, 1, 2, 2}, off[] = {1, 2};
    const size_t reversed[] = {1, 0};
    NsatCoverInput input = {.on = on, .off = off, .inputs = 2, .positives = 2,
        .negatives = 1, .target = 1, .row_order = reversed, .order_count = 2};
    NsatVerifier *v;
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited) == NSAT_OK);
    NsatSolver *s;
    NsatStopReason stop;
    assert(nsat_cover_load_solver(&s, &input, v, unlimited, &stop) == NSAT_ERROR);
    assert(stop == NSAT_STOP_PROOF);
    nsat_destroy(s);
    nsat_verify_destroy(v);
}

static void test_solver_cannot_replace_premises(void)
{
    NsatSolver *s;
    NsatVerifier *v;
    assert(nsat_init(&s, 2, unlimited) == NSAT_OK);
    assert(nsat_verify_init_cnf(&v, 2, unlimited) == NSAT_OK);
    add_pair(s, v, (int[]){1}, 1);
    assert(nsat_add_original(s, (int[]){-1}, 1) == NSAT_OK);
    assert(nsat_verify_add_original(v, (int[]){-1, 2}, 2) == NSAT_OK);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    assert(nsat_propagate_root(s, v) == NSAT_ERROR);
    assert(nsat_stats(s).stop == NSAT_STOP_PROOF);
    nsat_destroy(s);
    nsat_verify_destroy(v);
}

static uint32_t random_word(uint32_t *seed)
{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

static void test_seeded_search(void)
{
    uint32_t seed = UINT32_C(0x5a401fc);
    for (unsigned sample = 0; sample < 4096; ++sample) {
        int literals[72][5];
        ClauseView clauses[72];
        int n = 4 + (int)(random_word(&seed) % 11);
        size_t count = 1 + random_word(&seed) % 72;
        for (size_t c = 0; c < count; ++c) {
            size_t width = 2 + random_word(&seed) % (n > 4 ? 4 : 3);
            unsigned used = 0;
            for (size_t i = 0; i < width; ++i) {
                unsigned var;
                do { var = 1 + random_word(&seed) % (unsigned)n; }
                while (used & (1u << var));
                used |= 1u << var;
                literals[c][i] = random_word(&seed) & 1 ? (int)var : -(int)var;
            }
            clauses[c] = (ClauseView){literals[c], width};
        }
        NsatSearchOptions options = {
            .restart_base = sample % 2 ? 1 : 64,
            .disable_minimisation = sample % 4 == 3,
            .reduction_interval = 3
        };
        compare_search(clauses, count, n, options);
    }
    compare_search(NULL, 0, 0, (NsatSearchOptions){0});
    compare_search(NULL, 0, 7, (NsatSearchOptions){0});
}

static void pigeonhole(NsatSolver **s, NsatVerifier **v, int pigeons, int holes,
                        NsatLimits limits)
{
    int variables = pigeons * holes;
    assert(nsat_init(s, variables, limits) == NSAT_OK);
    assert(nsat_verify_init_cnf(v, variables, unlimited) == NSAT_OK);
    for (int p = 0; p < pigeons; ++p) {
        int choices[8];
        assert(holes <= 8);
        for (int h = 0; h < holes; ++h) choices[h] = p * holes + h + 1;
        add_pair(*s, *v, choices, (size_t)holes);
        for (int q = 0; q < p; ++q)
            for (int h = 0; h < holes; ++h)
                add_pair(*s, *v, (int[]){-choices[h], -(q * holes + h + 1)}, 2);
    }
    assert(nsat_verify_seal_originals(*v) == NSAT_OK);
}

static void test_learning(void)
{
    NsatSolver *s;
    NsatVerifier *v;
    NsatSearchOptions options = {.restart_base = 1, .reduction_interval = 1};
    pigeonhole(&s, &v, 6, 5, unlimited);
    assert(nsat_solve(s, v, options) == NSAT_UNSAT);
    NsatSearchStats stats = nsat_search_stats(s);
    assert(stats.binary_propagations && stats.restarts && stats.minimised_literals &&
           stats.nonchronological_backjumps && stats.conflicts == stats.learned);
    printf("Checked conflict stress: %llu learned, %llu minimised literals, %llu restarts, "
           "%llu nonchronological backjumps\n",
           (unsigned long long)stats.learned, (unsigned long long)stats.minimised_literals,
           (unsigned long long)stats.restarts, (unsigned long long)stats.nonchronological_backjumps);
    assert(stats.reductions && stats.deleted && stats.compactions &&
           stats.words_reclaimed && stats.moved);
    assert(nsat_verify_stats(v).clauses_deleted == stats.deleted);
    assert(nsat_solve(s, v, options) == NSAT_ERROR);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    pigeonhole(&s, &v, 5, 4, (NsatLimits){.conflicts = 1});
    assert(nsat_solve(s, v, options) == NSAT_UNKNOWN);
    assert(nsat_stats(s).stop == NSAT_STOP_CONFLICTS && nsat_search_stats(s).learned == 1);
    assert(nsat_model_value(s, 1) == 0);
    assert(nsat_solve(s, v, options) == NSAT_UNKNOWN);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    pigeonhole(&s, &v, 5, 5, unlimited);
    assert(nsat_solve(s, v, options) == NSAT_SAT);
    assert(nsat_add_original(s, NULL, 0) == NSAT_ERROR);
    assert(nsat_model_value(s, 1) == 0);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    /* A total model for a different formula must be rejected. */
    assert(nsat_init(&s, 2, unlimited) == NSAT_OK);
    assert(nsat_verify_init_cnf(&v, 2, unlimited) == NSAT_OK);
    assert(nsat_add_original(s, (int[]){-1, -2}, 2) == NSAT_OK);
    assert(nsat_verify_add_original(v, (int[]){1, 2}, 2) == NSAT_OK);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    assert(nsat_solve(s, v, options) == NSAT_ERROR);
    assert(nsat_stats(s).stop == NSAT_STOP_PROOF);
    assert(nsat_model_value(s, 1) == 0);
    nsat_destroy(s);
    nsat_verify_destroy(v);
}

static void test_plateau_counterexample(void)
{
    const char *rows[] = {
        "01100000", "10100000", "11000000", "11100000", "00011000",
        "00010000", "00001000", "00000100", "00000010", "00000001"
    };
    int points[80];
    for (size_t i = 0; i < 10; ++i)
        for (size_t j = 0; j < 8; ++j)
            points[8 * i + j] = 1 + rows[i][j] - '0';

    for (int k = 1; k <= 2; ++k) {
        NsatVerifier *v;
        NsatSolver *s;
        assert(nsat_verify_init_cover(&v, points, points + 40, 8, 5, 5, (size_t)k,
                                      NULL, 0, unlimited) == NSAT_OK);
        NsatCoverInput input = {.on = points, .off = points + 40, .inputs = 8,
            .positives = 5, .negatives = 5, .target = (size_t)k};
        NsatStopReason stop;
        assert(nsat_cover_load_solver(&s, &input, v, unlimited, &stop) == NSAT_OK);
        assert(nsat_solve(s, v, (NsatSearchOptions){0}) == (k == 1 ? NSAT_UNSAT : NSAT_SAT));
        if (k == 2) {
            for (size_t i = 0; i < 10; ++i)
                assert(model_covers(s, points + i * 8, 8, k) == (i < 5));
        }
        nsat_destroy(s);
        nsat_verify_destroy(v);
    }
}

static bool cancel(void *arg) { return *(bool *)arg; }

static bool cancel_search(void *arg)
{
    NsatSolver *s = *(NsatSolver **)arg;
    return s && nsat_search_stats(s).conflicts >= 3;
}

static void test_limits(void)
{
    NsatVerifier *v;
    bool stopped = false;
    NsatLimits limits = {.poll = cancel, .context = &stopped};
    assert(nsat_verify_init_cnf(&v, 1, limits) == NSAT_OK);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    stopped = true;
    assert(nsat_verify_derive(v, 1, NULL, 0, NULL, 0) == NSAT_UNKNOWN);
    stopped = false;
    assert(nsat_verify_conclude_unsat(v, 1) == NSAT_UNKNOWN);
    assert(nsat_verify_stats(v).stop == NSAT_STOP_CANCELLED);
    nsat_verify_destroy(v);

    assert(nsat_verify_init_cnf(&v, 4, (NsatLimits){.work = 1}) == NSAT_UNKNOWN);
    assert(nsat_verify_stats(v).stop == NSAT_STOP_WORK);
    nsat_verify_destroy(v);
    assert(nsat_verify_init_cnf(&v, 1, (NsatLimits){.memory_bytes = 1}) == NSAT_UNKNOWN);
    assert(!v);

    NsatSolver *s = NULL;
    pigeonhole(&s, &v, 6, 5, (NsatLimits){.poll = cancel_search, .context = &s});
    assert(nsat_solve(s, v, (NsatSearchOptions){0}) == NSAT_UNKNOWN);
    assert(nsat_stats(s).stop == NSAT_STOP_CANCELLED && nsat_search_stats(s).learned >= 2);
    assert(nsat_model_value(s, 1) == 0);
    nsat_destroy(s);
    nsat_verify_destroy(v);

    /* Both participants have loaded complete formulas when the shared
     * cap fires. Partial setup must leave no charged memory behind. */
    const int on[] = {1, 1, 2, 2}, off[] = {1, 2};
    NsatCoverInput input = {.on = on, .off = off, .inputs = 2, .positives = 2,
        .negatives = 1, .target = 2};
    for (int memory = 0; memory <= 1; ++memory) {
        NsatResources resources = {0};
        limits = (NsatLimits){.shared = &resources};
        assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 2, NULL, 0, limits) == NSAT_OK);
        NsatStopReason stop;
        assert(nsat_cover_load_solver(&s, &input, v, limits, &stop) == NSAT_OK);
        if (memory) resources.memory_limit = resources.memory + 1;
        else resources.work_limit = resources.work + 1;
        assert(nsat_solve(s, v, (NsatSearchOptions){0}) == NSAT_UNKNOWN);
        assert(resources.stop == (memory ? NSAT_STOP_MEMORY : NSAT_STOP_WORK));
        assert(nsat_model_value(s, 1) == 0);
        nsat_destroy(s);
        nsat_verify_destroy(v);
        assert(resources.memory == 0);
    }
}

#ifdef NSAT_TESTING
static NsatStatus allocate_through_root(void)
{
    NsatVerifier *v;
    NsatSolver *s = NULL;
    const int on[] = {1, 1, 2, 2}, off[] = {1, 2};
    NsatStatus result = nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited);
    if (result == NSAT_OK) result = nsat_init(&s, nsat_verify_variables(v), unlimited);
    for (NsatClauseId id = 1; result == NSAT_OK && id <= nsat_verify_original_count(v); ++id) {
        const int *lits;
        size_t count;
        assert(nsat_verify_clause(v, id, &lits, &count));
        result = nsat_add_original(s, lits, count);
    }
    if (result == NSAT_OK) result = nsat_propagate_root(s, v);
    if (result == NSAT_ERROR)
        assert((s ? nsat_stats(s) : v ? nsat_verify_stats(v) : (NsatStats){.stop = NSAT_STOP_ALLOCATION})
               .stop == NSAT_STOP_ALLOCATION || (s == NULL && v));
    nsat_destroy(s);
    nsat_verify_destroy(v);
    return result;
}

static void test_allocation_failures(void)
{
    long failed = 0;
    for (; failed < 256; ++failed) {
        allocations_before_failure = failed;
        NsatStatus result = allocate_through_root();
        if (result == NSAT_UNSAT) break;
        assert(result == NSAT_ERROR);
    }
    assert(failed > 20 && failed < 256);
    allocations_before_failure = -1;

    for (failed = 0; failed < 256; ++failed) {
        NsatSolver *s;
        NsatVerifier *v;
        pigeonhole(&s, &v, 4, 3, unlimited);
        allocations_before_failure = failed;
        NsatStatus result = nsat_solve(s, v, (NsatSearchOptions){.restart_base = 1});
        allocations_before_failure = -1;
        if (result != NSAT_UNSAT) {
            assert(result == NSAT_ERROR && nsat_stats(s).stop == NSAT_STOP_ALLOCATION);
            assert(nsat_model_value(s, 1) == 0);
        }
        nsat_destroy(s);
        nsat_verify_destroy(v);
        if (result == NSAT_UNSAT) break;
    }
    assert(failed > 20 && failed < 256);
}
#endif

int main(void)
{
    test_root();
    test_seeded_search();
    test_learning();
    test_cover();
    test_plateau_counterexample();
    test_domain_lemmas();
    test_incremental_lifecycle();
    test_native20_gate();
    test_encoder_rejects_different_originals();
    test_cover_validation();
    test_solver_cannot_replace_premises();
    test_limits();
#ifdef NSAT_TESTING
    test_allocation_failures();
#endif
    return 0;
}
