/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
/* The shared allowance covers matrix payloads, stack links and task copies. */
#include <stdatomic.h>

typedef struct {
    atomic_size_t used, peak;
    size_t capacity;
} LiveBases;

typedef struct SavedBasis {
    struct SavedBasis *parent;
    size_t bytes;
    double matrix[];
} SavedBasis;

typedef struct {
    double *projection, *prices;
    LiveBases *live;
    SavedBasis *top;
} CompactPricing;

static SavedBasis *reserve_basis(LiveBases *live, size_t rows)
{
    size_t bytes = sizeof(SavedBasis) + rows * rows * sizeof(double);
    size_t used = atomic_load_explicit(&live->used, memory_order_relaxed);
    do {
        if (bytes > live->capacity - used) return NULL;
    } while (!atomic_compare_exchange_weak_explicit(&live->used, &used, used + bytes,
                  memory_order_relaxed, memory_order_relaxed));
    SavedBasis *saved = malloc(bytes);
    if (!saved) {
        atomic_fetch_sub_explicit(&live->used, bytes, memory_order_relaxed);
        return NULL;
    }
    size_t peak = atomic_load_explicit(&live->peak, memory_order_relaxed);
    while (used + bytes > peak &&
           !atomic_compare_exchange_weak_explicit(&live->peak, &peak, used + bytes,
               memory_order_relaxed, memory_order_relaxed)) {}
    saved->bytes = bytes;
    return saved;
}

static void release_basis(LiveBases *live, SavedBasis *saved)
{
    if (!saved) return;
    size_t bytes = saved->bytes;
    free(saved);
    atomic_fetch_sub_explicit(&live->used, bytes, memory_order_relaxed);
}

static void restore_bases(CompactPricing *pricing, SavedBasis *mark)
{
    while (pricing->top != mark) {
        SavedBasis *saved = pricing->top;
        pricing->top = saved->parent;
        release_basis(pricing->live, saved);
    }
}

static void copy_basis_matrix(SavedBasis *saved, PricingBasis *basis)
{
    size_t rows = basis->rows;
    for (size_t row = 0; row < rows; ++row)
        memcpy(saved->matrix + row * rows, basis->inverse + row * basis->stride,
               rows * sizeof(double));
    basis->inverse = saved->matrix;
    basis->stride = rows;
}

static bool keep_projected_basis(NsatBudget *budget, CompactPricing *storage,
                                 PricingBasis *basis)
{
    if (!nsat_work(budget, basis->rows * basis->rows)) return false;
    SavedBasis *saved = reserve_basis(storage->live, basis->rows);
    if (!saved) return false;
    copy_basis_matrix(saved, basis);
    saved->parent = storage->top;
    storage->top = saved;
    return true;
}
