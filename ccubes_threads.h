#ifndef CCUBES_THREADS_H
#define CCUBES_THREADS_H

#include <stdbool.h>
#include <stdint.h>

#if defined(HAVE_OPENMP)
    #include <omp.h>
#elif defined(HAVE_PTHREAD)
    #include <pthread.h>
#endif

typedef void (*ccubes_range_worker)(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
);

int ccubes_default_thread_count(void);
int ccubes_effective_thread_count(int requested_threads);
void ccubes_set_thread_count(int threads);
int ccubes_thread_count(void);
const char *ccubes_thread_backend(void);

int ccubes_parallel_for(
    uint64_t start,
    uint64_t end,
    int requested_threads,
    bool strided,
    ccubes_range_worker worker,
    void *data
);

typedef struct {
#if defined(HAVE_OPENMP)
    omp_lock_t lock;
#elif defined(HAVE_PTHREAD)
    pthread_mutex_t mutex;
#else
    int unused;
#endif
} ccubes_mutex;

int ccubes_mutex_init(ccubes_mutex *mutex);
void ccubes_mutex_lock(ccubes_mutex *mutex);
void ccubes_mutex_unlock(ccubes_mutex *mutex);
void ccubes_mutex_destroy(ccubes_mutex *mutex);

#endif
