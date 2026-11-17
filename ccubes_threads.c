/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "ccubes_threads.h"
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <stdatomic.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

static int ccubes_configured_threads = 0;

typedef struct {
    ccubes_range_worker worker;
    void *data;
    uint64_t start;
    uint64_t end;
    uint64_t stride;
    int worker_id;
    int worker_count;
} CCubesRangeTask;

#if defined(HAVE_PTHREAD) && !defined(HAVE_OPENMP)
static void *ccubes_thread_main(void *arg) {
    CCubesRangeTask *task = (CCubesRangeTask *)arg;
    task->worker(
        task->start,
        task->end,
        task->stride,
        task->worker_id,
        task->worker_count,
        task->data
    );
    return NULL;
}
#endif

const char *ccubes_thread_backend(void) {
#if defined(HAVE_OPENMP)
    return "OpenMP";
#elif defined(HAVE_PTHREAD)
    return "pthread";
#else
    return "serial";
#endif
}

int ccubes_available_thread_count(void) {
    /* Capture the process allocation before OpenMP binds its master thread.
       Re-reading that thread's affinity after a parallel region can report
       one CPU even though the process has a much larger allocation. */
    static atomic_int allocation = 0;
    int cached = atomic_load_explicit(&allocation, memory_order_relaxed);
    if (cached > 0) return cached;
    int available = 1;
#if defined(HAVE_OPENMP) || defined(HAVE_PTHREAD)
#if defined(_WIN32)
    DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    available = count > INT_MAX ? INT_MAX : (int)count;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count > 0) available = count > INT_MAX ? INT_MAX : (int)count;
#if defined(__linux__)
    /* Linux affinity masks can exceed cpu_set_t on large hosts. Grow the
       byte buffer until the kernel accepts it; never assume CPU_SETSIZE. */
    for (size_t bytes = sizeof(cpu_set_t); bytes <= (size_t)INT_MAX / 8; bytes *= 2) {
        unsigned char *mask = calloc(bytes, 1);
        if (!mask) break;
        int rc = sched_getaffinity(0, bytes, (cpu_set_t *)mask);
        int error = errno;
        if (rc == 0) {
            int allowed = 0;
            for (size_t i = 0; i < bytes; ++i) {
                unsigned char bits = mask[i];
                while (bits) { ++allowed; bits &= bits - 1; }
            }
            if (allowed > 0) available = allowed;
        }
        free(mask);
        if (rc == 0 || error != EINVAL) break;
    }
#endif
#endif
#if defined(HAVE_OPENMP)
    int runtime_limit = omp_get_thread_limit();
    if (runtime_limit > 0 && available > runtime_limit) available = runtime_limit;
#endif

#endif
    if (available < 1) available = 1;
    int unset = 0;
    atomic_compare_exchange_strong_explicit(&allocation, &unset, available,
        memory_order_relaxed, memory_order_relaxed);
    return unset > 0 ? unset : available;
}

int ccubes_default_thread_count(void) {
    int available = ccubes_available_thread_count();
#if defined(HAVE_OPENMP)
    int default_team = omp_get_max_threads();
    if (default_team > 0 && default_team < available) available = default_team;
#endif
    return available;
}

int ccubes_effective_thread_count(int requested_threads) {
    int threads = requested_threads;
    if (threads <= 0)
        threads = ccubes_configured_threads > 0 ? ccubes_configured_threads : ccubes_default_thread_count();
    int available = ccubes_available_thread_count();
    if (threads > available) threads = available;
    return threads > 0 ? threads : 1;
}

uint64_t ccubes_monotonic_nanoseconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER frequency, now;
    if (!QueryPerformanceFrequency(&frequency) || !QueryPerformanceCounter(&now)) return 0;
    return (uint64_t)((long double)now.QuadPart * 1000000000.0L / frequency.QuadPart);
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
#endif
}

int ccubes_adaptive_team_choose(ccubes_adaptive_team *team, uint64_t items,
                               uint64_t work, int requested) {
    int limit = requested > 0 ? requested : 1;
    if ((uint64_t)limit > items) limit = items > 0 ? (int)items : 1;
    uint64_t work_class = 0;
    for (uint64_t size = work; size; size >>= 1) ++work_class;
    if (!team->best || team->limit != limit || team->work_class != work_class) {
        *team = (ccubes_adaptive_team){.limit = limit, .work_class = work_class,
            .best = 1, .candidate = 1, .settled = limit == 1};
    }
    return team->settled ? team->best : team->candidate;
}

void ccubes_adaptive_team_observe(ccubes_adaptive_team *team, uint64_t elapsed) {
    if (team->settled || !elapsed) return;
    team->samples[team->sample_count++] = elapsed;
    if (team->sample_count < 3) return;
    uint64_t a = team->samples[0], b = team->samples[1], c = team->samples[2];
    uint64_t median = a < b ? (b < c ? b : (a < c ? c : a))
                            : (a < c ? a : (b < c ? c : b));
    team->sample_count = 0;
    /* Demand a repeatable 10% gain, avoiding oscillation on timer noise.
       Three real calls amortize each trial; no additional search is done. */
    if (team->candidate == 1 || median < team->best_nanoseconds - team->best_nanoseconds / 10) {
        team->best = team->candidate;
        team->best_nanoseconds = median;
        if (team->candidate == team->limit) team->settled = true;
        else team->candidate = team->candidate > team->limit / 2 ? team->limit : team->candidate * 2;
    } else {
        team->settled = true;
    }
}

void ccubes_set_thread_count(int threads) {
    ccubes_configured_threads = ccubes_effective_thread_count(threads);
#if defined(HAVE_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(ccubes_configured_threads);
#endif
}

int ccubes_thread_count(void) {
    return ccubes_effective_thread_count(0);
}

int ccubes_parallel_for(
    uint64_t start,
    uint64_t end,
    int requested_threads,
    bool strided,
    ccubes_range_worker worker,
    void *data
) {
    return ccubes_parallel_for_counted(start, end, requested_threads, strided,
        worker, data, NULL);
}

int ccubes_parallel_for_counted(
    uint64_t start, uint64_t end, int requested_threads, bool strided,
    ccubes_range_worker worker, void *data, int *actual_threads
) {
    if (actual_threads) *actual_threads = 0;
    if (!worker || end <= start) {
        return 1;
    }

    uint64_t count = end - start;
    int nthreads = ccubes_effective_thread_count(requested_threads);
    if ((uint64_t)nthreads > count) {
        nthreads = (int)count;
    }
    if (nthreads <= 1) {
        worker(start, end, 1, 0, 1, data);
        if (actual_threads) *actual_threads = 1;
        return 1;
    }

#if defined(HAVE_OPENMP)
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int workers = omp_get_num_threads();
        if (tid == 0 && actual_threads) *actual_threads = workers;
        if (strided) {
            worker(start + (uint64_t)tid, end, (uint64_t)workers, tid, workers, data);
        } else {
            uint64_t width = count / (uint64_t)workers;
            uint64_t rem = count % (uint64_t)workers;
            uint64_t chunk_start = start + (uint64_t)tid * width + ((uint64_t)tid < rem ? (uint64_t)tid : rem);
            uint64_t chunk_width = width + ((uint64_t)tid < rem ? 1u : 0u);
            worker(chunk_start, chunk_start + chunk_width, 1, tid, workers, data);
        }
    }
    return 1;
#elif defined(HAVE_PTHREAD)
    pthread_t *threads = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
    CCubesRangeTask *tasks = (CCubesRangeTask *)calloc((size_t)nthreads, sizeof(CCubesRangeTask));
    if (!threads || !tasks) {
        free(threads);
        free(tasks);
        return 0;
    }

    const uint64_t chunk = count / (uint64_t)nthreads;
    const uint64_t rem = count % (uint64_t)nthreads;
    uint64_t next = start;

    /* Resolve every logical task before creating OS threads.  If pthread
     * creation later stops partway through, the caller thread can execute the
     * untouched suffix exactly once while the created prefix runs normally. */
    for (int i = 0; i < nthreads; i++) {
        tasks[i].worker = worker;
        tasks[i].data = data;
        tasks[i].worker_id = i;
        tasks[i].worker_count = nthreads;

        if (strided) {
            tasks[i].start = start + (uint64_t)i;
            tasks[i].end = end;
            tasks[i].stride = (uint64_t)nthreads;
        } else {
            uint64_t width = chunk + ((uint64_t)i < rem ? 1u : 0u);
            tasks[i].start = next;
            tasks[i].end = next + width;
            tasks[i].stride = 1;
            next += width;
        }
    }

    int started = 0;
    for (int i = 0; i < nthreads; ++i) {
        if (pthread_create(&threads[i], NULL, ccubes_thread_main, &tasks[i]) != 0) {
            break;
        }
        started++;
    }

    for (int i = started; i < nthreads; ++i) {
        ccubes_thread_main(&tasks[i]);
    }

    for (int i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
    }
    if (actual_threads) *actual_threads = started + (started < nthreads);

    free(threads);
    free(tasks);
    return 1;
#else
    (void)strided;
    worker(start, end, 1, 0, 1, data);
    if (actual_threads) *actual_threads = 1;
    return 1;
#endif
}

int ccubes_mutex_init(ccubes_mutex *mutex) {
#if defined(HAVE_OPENMP)
    if (!mutex) return 0;
    omp_init_lock(&mutex->lock);
    return 1;
#elif defined(HAVE_PTHREAD)
    return mutex && pthread_mutex_init(&mutex->mutex, NULL) == 0;
#else
    (void)mutex;
    return 1;
#endif
}

void ccubes_mutex_lock(ccubes_mutex *mutex) {
#if defined(HAVE_OPENMP)
    if (mutex) omp_set_lock(&mutex->lock);
#elif defined(HAVE_PTHREAD)
    if (mutex) pthread_mutex_lock(&mutex->mutex);
#else
    (void)mutex;
#endif
}

void ccubes_mutex_unlock(ccubes_mutex *mutex) {
#if defined(HAVE_OPENMP)
    if (mutex) omp_unset_lock(&mutex->lock);
#elif defined(HAVE_PTHREAD)
    if (mutex) pthread_mutex_unlock(&mutex->mutex);
#else
    (void)mutex;
#endif
}

void ccubes_mutex_destroy(ccubes_mutex *mutex) {
#if defined(HAVE_OPENMP)
    if (mutex) omp_destroy_lock(&mutex->lock);
#elif defined(HAVE_PTHREAD)
    if (mutex) pthread_mutex_destroy(&mutex->mutex);
#else
    (void)mutex;
#endif
}

int ccubes_condition_init(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    if (pthread_mutex_init(&condition->mutex, NULL) != 0) return 0;
    if (pthread_cond_init(&condition->changed, NULL) != 0) {
        pthread_mutex_destroy(&condition->mutex);
        return 0;
    }
    return 1;
#else
    return ccubes_mutex_init(&condition->mutex);
#endif
}

void ccubes_condition_lock(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    pthread_mutex_lock(&condition->mutex);
#else
    ccubes_mutex_lock(&condition->mutex);
#endif
}

void ccubes_condition_unlock(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    pthread_mutex_unlock(&condition->mutex);
#else
    ccubes_mutex_unlock(&condition->mutex);
#endif
}

void ccubes_condition_wait(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    pthread_cond_wait(&condition->changed, &condition->mutex);
#else
    /* OpenMP supplies a lock but has no condition variable. */
    ccubes_condition_unlock(condition);
#if defined(HAVE_OPENMP) && defined(_WIN32)
    Sleep(1);
#elif defined(HAVE_OPENMP)
    nanosleep(&(struct timespec){.tv_nsec = 100000}, NULL);
#endif
    ccubes_condition_lock(condition);
#endif
}

void ccubes_condition_broadcast(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    pthread_cond_broadcast(&condition->changed);
#else
    (void)condition;
#endif
}

void ccubes_condition_destroy(ccubes_condition *condition) {
#if defined(HAVE_PTHREAD)
    pthread_cond_destroy(&condition->changed);
    pthread_mutex_destroy(&condition->mutex);
#else
    ccubes_mutex_destroy(&condition->mutex);
#endif
}
