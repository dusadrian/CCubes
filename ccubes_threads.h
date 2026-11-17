#ifndef CCUBES_THREADS_H
#define CCUBES_THREADS_H

#include <stdbool.h>
#include <stdint.h>

#if defined(HAVE_OPENMP)
    #include <omp.h>
#endif
#if defined(HAVE_PTHREAD)
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
int ccubes_available_thread_count(void);
int ccubes_effective_thread_count(int requested_threads);
void ccubes_set_thread_count(int threads);
int ccubes_thread_count(void);
const char *ccubes_thread_backend(void);

/* One controller per independent kernel/workload class. Observations include
 * dispatch and join; callers execute real work only, never benchmark replays. */
typedef struct {
    uint64_t work_class, samples[3], best_nanoseconds;
    int limit, best, candidate, sample_count;
    bool settled;
} ccubes_adaptive_team;
int ccubes_adaptive_team_choose(ccubes_adaptive_team *team, uint64_t items,
                               uint64_t work, int requested);
void ccubes_adaptive_team_observe(ccubes_adaptive_team *team, uint64_t elapsed);
uint64_t ccubes_monotonic_nanoseconds(void);

int ccubes_parallel_for(
    uint64_t start,
    uint64_t end,
    int requested_threads,
    bool strided,
    ccubes_range_worker worker,
    void *data
);

/* Same dispatch, with the number of executors that ran callbacks. */
int ccubes_parallel_for_counted(
    uint64_t start, uint64_t end, int requested_threads, bool strided,
    ccubes_range_worker worker, void *data, int *actual_threads
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

typedef struct {
#if defined(HAVE_PTHREAD)
    pthread_mutex_t mutex;
    pthread_cond_t changed;
#else
    ccubes_mutex mutex;
#endif
} ccubes_condition;

int ccubes_condition_init(ccubes_condition *condition);
void ccubes_condition_lock(ccubes_condition *condition);
void ccubes_condition_unlock(ccubes_condition *condition);
/* Call with the lock held and recheck the predicate after waking. */
void ccubes_condition_wait(ccubes_condition *condition);
void ccubes_condition_broadcast(ccubes_condition *condition);
void ccubes_condition_destroy(ccubes_condition *condition);

#endif
