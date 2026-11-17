#include "native_sat_support.h"

typedef struct {
    NsatClauseId id;
    size_t count;
    bool target_bound;
    int lits[];
} Clause;

/* Marks a hash slot whose derived clause was deleted. */
static Clause deleted;

/* Originals have the dense IDs 1..originals and are indexed directly.
 * Derived clauses use the hash table, where a NULL slot was never used. */
struct NsatVerifier {
    NsatBudget budget;
    int variables;
    NsatClauseId last_id, originals;
    enum { LOADING, SEALED, SAT_CONCLUDED, CONCLUDED } phase;
    Clause **loaded;
    size_t loaded_capacity, loaded_literals;
    Clause **table;
    size_t capacity, derived;
    union {
        size_t original_words; /* Used until the first derived table is built. */
        size_t occupied;
    };
    signed char *values;
    int *assigned;
    size_t assigned_count;
    bool started;
    size_t cover_slots, cover_target;
};

static size_t hash(NsatClauseId id)
{
    id ^= id >> 33;
    id *= UINT64_C(0xff51afd7ed558ccd);
    return (size_t)(id ^ (id >> 33));
}

static Clause **find_derived(const NsatVerifier *v, NsatClauseId id, NsatBudget *budget)
{
    if (id <= v->originals || !v->capacity) return NULL;
    size_t i = hash(id) & (v->capacity - 1);
    while (!budget || nsat_work(budget, 1)) {
        Clause **slot = &v->table[i];
        if (!*slot) return NULL;
        if (*slot != &deleted && (*slot)->id == id) return slot;
        i = (i + 1) & (v->capacity - 1);
    }
    return NULL;
}

static Clause *lookup(const NsatVerifier *v, NsatClauseId id, NsatBudget *budget)
{
    if (!id) return NULL;
    if (id <= v->originals) {
        if (budget && !nsat_work(budget, 1)) return NULL;
        return v->loaded[id - 1];
    }
    Clause **slot = find_derived(v, id, budget);
    return slot ? *slot : NULL;
}

static bool insert(NsatBudget *budget, Clause **table, size_t capacity,
                   Clause *clause, size_t *occupied)
{
    size_t i = hash(clause->id) & (capacity - 1);
    while (nsat_work(budget, 1)) {
        if (!table[i] || table[i] == &deleted) {
            if (!table[i]) ++*occupied;
            table[i] = clause;
            return true;
        }
        i = (i + 1) & (capacity - 1);
    }
    return false;
}

static bool rebuild(NsatVerifier *v, size_t capacity)
{
    NsatBudget *b = &v->budget;
    Clause **table = nsat_alloc(b, capacity, sizeof(*table));
    if (!table) return false;
    size_t occupied = 0;
    if (!nsat_copy(b, table, NULL, capacity * sizeof(*table))) goto fail;
    for (size_t i = 0; i < v->capacity; ++i) {
        if (!nsat_work(b, 1)) goto fail;
        Clause *clause = v->table[i];
        if (clause && clause != &deleted && !insert(b, table, capacity, clause, &occupied))
            goto fail;
    }
    nsat_free(b, v->table, v->capacity * sizeof(*v->table));
    v->table = table;
    v->capacity = capacity;
    v->occupied = occupied;
    ++b->stats.index_rebuilds;
    return true;
fail:
    nsat_free(b, table, capacity * sizeof(*table));
    return false;
}

/* Make room for this many originals in one step. The count is a sizing
 * hint: loading more only resumes the ordinary doubling. */
static bool reserve_originals(NsatVerifier *v, size_t count)
{
    NsatBudget *b = &v->budget;
    if (count <= v->loaded_capacity) return true;
    Clause **loaded = nsat_alloc(b, count, sizeof(*loaded));
    if (!loaded) return false;
    if (!nsat_copy(b, loaded, v->loaded, (size_t)v->originals * sizeof(*loaded))) {
        nsat_free(b, loaded, count * sizeof(*loaded));
        return false;
    }
    nsat_free(b, v->loaded, v->loaded_capacity * sizeof(*loaded));
    v->loaded = loaded;
    v->loaded_capacity = count;
    return true;
}

static bool reserve_derived(NsatVerifier *v)
{
    if (v->capacity && v->occupied < v->capacity / 2) return true;
    size_t capacity = v->capacity ? v->capacity : 16;
    if (v->derived >= capacity / 2) {
        if (!nsat_mul(capacity, 2, &capacity))
            return nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_INPUT);
    }
    return rebuild(v, capacity);
}

static bool save(NsatVerifier *v, NsatClauseId id, const int *lits, size_t count)
{
    NsatBudget *b = &v->budget;
    const bool original = v->phase == LOADING;
    if (original) {
        if (v->originals == v->loaded_capacity) {
            size_t capacity = 8;
            if (v->loaded_capacity && !nsat_mul(v->loaded_capacity, 2, &capacity))
                return nsat_stop(b, NSAT_ERROR, NSAT_STOP_INPUT);
            if (!reserve_originals(v, capacity)) return false;
        }
    } else if (!reserve_derived(v)) {
        return false;
    }
    if (count > (SIZE_MAX - sizeof(Clause)) / sizeof(int))
        return nsat_stop(b, NSAT_ERROR, NSAT_STOP_INPUT);
    size_t bytes = sizeof(Clause) + count * sizeof(int);
    Clause *clause = nsat_alloc(b, bytes, 1);
    if (!clause) return false;
    clause->id = id;
    clause->count = count;
    clause->target_bound = false;
    if (!nsat_copy(b, clause->lits, lits, count * sizeof(int)) ||
        !nsat_poll(b) ||
        (!original && !insert(b, v->table, v->capacity, clause, &v->occupied))) {
        nsat_free(b, clause, bytes);
        return false;
    }
    if (original) {
        v->loaded[id - 1] = clause;
        v->loaded_literals += count;
        v->original_words = nsat_clause_storage_capacity(v->original_words, v->loaded_literals);
    } else {
        ++v->derived;
    }
    v->last_id = id;
    return true;
}

NsatStatus nsat_verify_init_cnf(NsatVerifier **out, int variables, NsatLimits limits)
{
    if (!out) return NSAT_ERROR;
    *out = NULL;
    if (variables < 0) return NSAT_ERROR;
    NsatBudget budget = {.limits = limits};
    NsatVerifier *v = nsat_alloc(&budget, 1, sizeof(*v));
    if (!v) return budget.status;
    *v = (NsatVerifier){.budget = budget, .variables = variables};
    *out = v;
    size_t count = (size_t)variables + 1;
    v->values = nsat_alloc(&v->budget, count, sizeof(*v->values));
    if (!v->values || !nsat_copy(&v->budget, v->values, NULL, count))
        return v->budget.status;
    v->assigned = nsat_alloc(&v->budget, count, sizeof(*v->assigned));
    return v->budget.status;
}

static bool require_phase(NsatVerifier *v, int phase)
{
    if (v->budget.status != NSAT_OK) return false;
    if ((int)v->phase != phase)
        return nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_STATE);
    return nsat_poll(&v->budget);
}

static void assign(NsatVerifier *v, int var, int value)
{
    v->values[var] = (signed char)value;
    v->assigned[v->assigned_count++] = var;
}

static bool assume_false(NsatVerifier *v, const int *lits, size_t count)
{
    if (count > (size_t)v->variables || (count && !lits))
        return nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_INPUT);
    for (size_t i = 0; i < count; ++i) {
        if (!nsat_work(&v->budget, 1)) return false;
        int var = nsat_var(lits[i], v->variables);
        if (!var || v->values[var])
            return nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_INPUT);
        assign(v, var, lits[i] < 0 ? 1 : -1);
    }
    return true;
}

static bool clear_values(NsatVerifier *v)
{
    while (v->assigned_count) {
        if (!nsat_work(&v->budget, 1)) return false;
        v->values[v->assigned[--v->assigned_count]] = 0;
    }
    return true;
}

NsatStatus nsat_verify_add_original(NsatVerifier *v, const int *lits, size_t count)
{
    if (!v) return NSAT_ERROR;
    if (!require_phase(v, LOADING)) return v->budget.status;
    if (v->last_id == UINT64_MAX)
        nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_INPUT);
    else if (assume_false(v, lits, count) && clear_values(v) &&
             save(v, v->last_id + 1, lits, count))
        v->originals = v->last_id;
    return v->budget.status;
}

NsatStatus nsat_verify_seal_originals(NsatVerifier *v)
{
    if (!v) return NSAT_ERROR;
    if (require_phase(v, LOADING)) v->phase = SEALED;
    return v->budget.status;
}

NsatStatus nsat_verify_begin(NsatVerifier *v)
{
    if (!v) return NSAT_ERROR;
    if (!require_phase(v, SEALED)) return v->budget.status;
    if (v->started || v->last_id != v->originals)
        nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_STATE);
    else
        v->started = true;
    return v->budget.status;
}

NsatStatus nsat_verify_derive(NsatVerifier *v, NsatClauseId id,
                              const int *lits, size_t count,
                              const NsatClauseId *hints, size_t hint_count)
{
    if (!v) return NSAT_ERROR;
    NsatBudget *b = &v->budget;
    if (!require_phase(v, SEALED)) return b->status;
    if (id <= v->last_id || !hints || !hint_count || (hint_count > v->derived && hint_count - v->derived > v->originals)) {
        nsat_stop(b, NSAT_ERROR, NSAT_STOP_PROOF);
        return b->status;
    }
    if (!assume_false(v, lits, count)) return b->status;

    for (size_t h = 0; h < hint_count; ++h) {
        const Clause *clause = lookup(v, hints[h], b);
        if (!clause) goto invalid;
        int unit = 0;
        for (size_t i = 0; i < clause->count; ++i) {
            if (!nsat_work(b, 1)) return b->status;
            int lit = clause->lits[i];
            int value = lit > 0 ? v->values[lit] : -v->values[-lit];
            if (value > 0 || (!value && unit)) goto invalid;
            if (!value) unit = lit;
        }
        if (h == hint_count - 1) {
            if (unit) goto invalid;
        } else {
            if (!unit) goto invalid;
            assign(v, unit > 0 ? unit : -unit, unit > 0 ? 1 : -1);
        }
    }
    if (clear_values(v) && save(v, id, lits, count)) ++b->stats.lemmas_checked;
    return b->status;
invalid:
    nsat_stop(b, NSAT_ERROR, NSAT_STOP_PROOF);
    return b->status;
}

NsatStatus nsat_verify_delete(NsatVerifier *v, NsatClauseId id)
{
    if (!v) return NSAT_ERROR;
    NsatBudget *b = &v->budget;
    if (!require_phase(v, SEALED)) return b->status;
    Clause **slot = find_derived(v, id, b);
    if (!slot || (*slot)->target_bound) {
        nsat_stop(b, NSAT_ERROR, NSAT_STOP_PROOF);
        return b->status;
    }
    if (!nsat_poll(b)) return b->status;
    nsat_free(b, *slot, sizeof(Clause) + (*slot)->count * sizeof(int));
    *slot = &deleted;
    --v->derived;
    ++b->stats.clauses_deleted;
    if (v->capacity > 16 && v->derived <= v->capacity / 8)
        rebuild(v, v->capacity / 2);
    return b->status;
}

NsatStatus nsat_verify_conclude_unsat(NsatVerifier *v, NsatClauseId id)
{
    if (!v) return NSAT_ERROR;
    if (!require_phase(v, SEALED)) return v->budget.status;
    const Clause *clause = lookup(v, id, &v->budget);
    if (!clause || clause->count)
        nsat_stop(&v->budget, NSAT_ERROR, NSAT_STOP_PROOF);
    if (!nsat_poll(&v->budget)) return v->budget.status;
    v->phase = CONCLUDED;
    return NSAT_UNSAT;
}

NsatStatus nsat_verify_conclude_sat(NsatVerifier *v, const int *model, size_t count)
{
    if (!v) return NSAT_ERROR;
    NsatBudget *b = &v->budget;
    if (!require_phase(v, SEALED)) return b->status;
    if (count != (size_t)v->variables || (count && !model)) goto invalid;
    for (size_t i = 0; i < count; ++i) {
        if (!nsat_work(b, 1)) return b->status;
        int lit = model[i];
        int var = nsat_var(lit, v->variables);
        if (!var || v->values[var]) goto invalid;
        assign(v, var, lit > 0 ? 1 : -1);
    }
    /* Every original, then each target unit among the derived clauses. */
    const size_t originals = (size_t)v->originals;
    for (size_t i = 0; i < originals + v->capacity; ++i) {
        if (!nsat_work(b, 1)) return b->status;
        const Clause *clause = i < originals ? v->loaded[i] : v->table[i - originals];
        if (!clause || clause == &deleted || (i >= originals && !clause->target_bound))
            continue;
        bool satisfied = false;
        for (size_t j = 0; j < clause->count; ++j) {
            if (!nsat_work(b, 1)) return b->status;
            int lit = clause->lits[j];
            if ((lit > 0 ? v->values[lit] : -v->values[-lit]) > 0) {
                satisfied = true;
                break;
            }
        }
        if (!satisfied) goto invalid;
    }
    if (!clear_values(v) || !nsat_poll(b)) return b->status;
    v->phase = SAT_CONCLUDED;
    return NSAT_SAT;
invalid:
    nsat_stop(b, NSAT_ERROR, NSAT_STOP_PROOF);
    return b->status;
}

NsatStatus nsat_verify_tighten_cover(NsatVerifier *v, size_t target,
                                      NsatClauseId *id, int *literal)
{
    if (!v) return NSAT_ERROR;
    NsatBudget *b = &v->budget;
    if (!require_phase(v, SAT_CONCLUDED)) return b->status;
    if (!v->started || !v->cover_slots || !target || target >= v->cover_target ||
        v->last_id == UINT64_MAX || !id || !literal) {
        nsat_stop(b, NSAT_ERROR, NSAT_STOP_STATE);
        return b->status;
    }
    /* The last K variable IDs are the final row's prefix-use variables.
     * Forbid zero-based slot target. First use forces all later slots off. */
    int unit = -(v->variables - (int)v->cover_slots + 1 + (int)target);
    NsatClauseId next = v->last_id + 1;
    if (!save(v, next, &unit, 1)) return b->status;
    Clause *clause = lookup(v, next, b);
    if (!clause) {
        nsat_stop(b, NSAT_ERROR, NSAT_STOP_STATE);
        return b->status;
    }
    clause->target_bound = true;
    v->cover_target = target;
    v->phase = SEALED;
    *id = next;
    *literal = unit;
    return NSAT_OK;
}

size_t nsat_verify_cover_target(const NsatVerifier *v) { return v->cover_target; }
size_t nsat_verify_cover_slots(const NsatVerifier *v) { return v->cover_slots; }

NsatStats nsat_verify_stats(const NsatVerifier *v) { return v->budget.stats; }
int nsat_verify_variables(const NsatVerifier *v) { return v->variables; }
NsatClauseId nsat_verify_original_count(const NsatVerifier *v) { return v->originals; }

bool nsat_verify_original_storage(const NsatVerifier *v, size_t *clauses, size_t *words)
{
    if (!v || !clauses || !words || v->phase != SEALED || v->started ||
        v->last_id != v->originals || v->budget.status != NSAT_OK) return false;
    *clauses = (size_t)v->originals;
    *words = v->original_words;
    return true;
}

bool nsat_verify_clause(const NsatVerifier *v, NsatClauseId id, const int **lits,
                        size_t *count)
{
    const Clause *clause = v->budget.status == NSAT_OK ? lookup(v, id, NULL) : NULL;
    if (!clause || !lits || !count) return false;
    *lits = clause->lits;
    *count = clause->count;
    return true;
}

void nsat_verify_destroy(NsatVerifier *v)
{
    if (!v) return;
    for (size_t i = 0; i < (size_t)v->originals; ++i) free(v->loaded[i]);
    free(v->loaded);
    for (size_t i = 0; i < v->capacity; ++i) {
        if (v->table[i] != &deleted) free(v->table[i]);
    }
    free(v->table);
    free(v->values);
    free(v->assigned);
    nsat_release_budget(&v->budget);
    free(v);
}

static bool valid_care(NsatVerifier *v, const int *on, const int *off,
                       size_t n, size_t on_count, size_t off_count)
{
    NsatBudget *b = &v->budget;
    size_t capacity = 16;
    while (off_count > capacity / 2) {
        if (!nsat_mul(capacity, 2, &capacity))
            return nsat_stop(b, NSAT_ERROR, NSAT_STOP_INPUT);
    }
    const int **table = nsat_alloc(b, capacity, sizeof(*table));
    if (!table) return false;
    bool ok = nsat_copy(b, table, NULL, capacity * sizeof(*table));
    for (size_t side = 0; ok && side < 2; ++side) {
        const int *points = side ? on : off;
        size_t rows = side ? on_count : off_count;
        for (size_t row = 0; ok && row < rows; ++row) {
            const int *point = points + row * n;
            uint64_t key = UINT64_C(14695981039346656037);
            for (size_t i = 0; i < n; ++i) {
                if (!nsat_work(b, 1)) goto done;
                if (point[i] != 1 && point[i] != 2) goto invalid;
                key = (key ^ (unsigned)point[i]) * UINT64_C(1099511628211);
            }
            size_t pos = hash(key) & (capacity - 1);
            while (table[pos]) {
                if (!nsat_work(b, 1)) goto done;
                size_t i = 0;
                while (i < n && table[pos][i] == point[i]) {
                    if (!nsat_work(b, 1)) goto done;
                    ++i;
                }
                if (i == n) {
                    if (side) goto invalid;
                    break;
                }
                pos = (pos + 1) & (capacity - 1);
            }
            if (!side) table[pos] = point;
        }
    }
    goto done;
invalid:
    nsat_stop(b, NSAT_ERROR, NSAT_STOP_INPUT);
done:
    nsat_free(b, table, capacity * sizeof(*table));
    return b->status == NSAT_OK;
}

/* Originals of a cover encoding: slots * (first + (rows - 1) * later + off),
 * where the arguments count the clauses of one slot for the first ON row,
 * for each later ON row, and for the OFF block. Zero when it does not fit. */
static size_t cover_original_count(size_t first, size_t later, size_t rows,
                                   size_t off, size_t slots) {
    size_t count;
    if (!nsat_mul(rows - 1, later, &count) || first > SIZE_MAX - count) return 0;
    count += first;
    if (off > SIZE_MAX - count || !nsat_mul(count + off, slots, &count)) return 0;
    return count;
}

/* Encoding v1. Number selector pairs by slot/condition, then assignment
 * and prefix variables by position/slot. Emit all ON blocks before OFF. */
static void cover_originals(NsatVerifier *v, const int *on, const int *off,
                            size_t n, size_t p, size_t q, size_t k,
                            const size_t *order, int *scratch)
{
    int selectors = (int)(2 * n * k);
    int prefix_offset = (int)(p * k);
    for (size_t r = 0; r < p; ++r) {
        const int *point = on + n * (order ? order[r] : r);
        int first = selectors + (int)(r * k) + 1;
        for (size_t t = 0; t < k; ++t) {
            if (!nsat_work(&v->budget, 1)) return;
            scratch[t] = first + (int)t;
        }
        if (nsat_verify_add_original(v, scratch, k) != NSAT_OK) return;
        for (size_t t = 0; t < k; ++t) {
            int a = first + (int)t;
            int u = a + prefix_offset;
            int base = (int)(2 * t * n);
            for (size_t i = 0; i < n; ++i) {
                int opposite = base + (int)(2 * i) + (3 - point[i]);
                if (nsat_verify_add_original(v, (int[]){-a, -opposite}, 2) != NSAT_OK)
                    return;
            }
            if (nsat_verify_add_original(v, (int[]){-a, u}, 2) != NSAT_OK)
                return;
            if (!r) {
                if (nsat_verify_add_original(v, (int[]){-u, a}, 2) != NSAT_OK)
                    return;
                if (t && nsat_verify_add_original(v, (int[]){-a}, 1) != NSAT_OK)
                    return;
            } else {
                int previous = u - (int)k;
                if (nsat_verify_add_original(v, (int[]){-previous, u}, 2) != NSAT_OK ||
                    nsat_verify_add_original(v, (int[]){-u, previous, a}, 3) != NSAT_OK)
                    return;
                if (t && nsat_verify_add_original(v, (int[]){-a, previous - 1}, 2) != NSAT_OK)
                    return;
            }
        }
    }
    for (size_t t = 0; t < k; ++t) {
        int base = (int)(2 * t * n);
        for (size_t i = 0; i < n; ++i) {
            int zero = base + (int)(2 * i) + 1;
            if (nsat_verify_add_original(v, (int[]){-zero, -(zero + 1)}, 2) != NSAT_OK)
                return;
        }
        for (size_t r = 0; r < q; ++r) {
            for (size_t i = 0; i < n; ++i) {
                if (!nsat_work(&v->budget, 1)) return;
                scratch[i] = base + (int)(2 * i) + (3 - off[r * n + i]);
            }
            if (nsat_verify_add_original(v, scratch, n) != NSAT_OK) return;
        }
    }
}

NsatStatus nsat_verify_init_cover(NsatVerifier **out, const int *on, const int *off,
                                  size_t n, size_t on_count, size_t off_count,
                                  size_t target, const size_t *row_order,
                                  size_t order_count, NsatLimits limits)
{
    if (!out) return NSAT_ERROR;
    *out = NULL;
    size_t cells;
    size_t half = (size_t)INT_MAX / 2;
    if (!on || !off || !n || !on_count || !off_count || !target ||
        (row_order ? order_count != on_count : order_count != 0) ||
        n > half || on_count > half - n || target > half / (n + on_count) ||
        !nsat_mul(on_count, n, &cells) || cells > SIZE_MAX / sizeof(int) ||
        !nsat_mul(off_count, n, &cells) || cells > SIZE_MAX / sizeof(int))
        return NSAT_ERROR;

    int variables = (int)(2 * (n + on_count) * target);
    NsatStatus status = nsat_verify_init_cnf(out, variables, limits);
    if (status != NSAT_OK) return status;
    NsatVerifier *v = *out;
    NsatBudget *b = &v->budget;
    if (row_order) {
        /* Use the otherwise empty scratch assignment array as a seen array. */
        for (size_t i = 0; i < on_count; ++i) {
            if (!nsat_work(b, 1)) return b->status;
            size_t row = row_order[i];
            if (row >= on_count || v->values[row]) {
                nsat_stop(b, NSAT_ERROR, NSAT_STOP_INPUT);
                return b->status;
            }
            v->values[row] = 1;
        }
        if (!nsat_copy(b, v->values, NULL, on_count)) return b->status;
    }
    if (!valid_care(v, on, off, n, on_count, off_count)) return b->status;
    if (!reserve_originals(v, cover_original_count(n + 3, n + 4, on_count,
                                                   n + off_count, target)))
        return b->status;
    size_t width = n > target ? n : target;
    int *scratch = nsat_alloc(b, width, sizeof(*scratch));
    if (!scratch) return b->status;
    cover_originals(v, on, off, n, on_count, off_count, target, row_order, scratch);
    nsat_free(b, scratch, width * sizeof(*scratch));
    v->cover_slots = v->cover_target = target;
    return nsat_verify_seal_originals(v);
}
