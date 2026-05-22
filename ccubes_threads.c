/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "ccubes_threads.h"

#include <stdlib.h>

#if defined(HAVE_PTHREAD) && !defined(HAVE_OPENMP)
    #if defined(_WIN32)
        #include <windows.h>
    #else
        #include <unistd.h>
    #endif
#endif

#define CCUBES_THREAD_LIMIT 64

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

int ccubes_default_thread_count(void) {
#if defined(HAVE_OPENMP)
    int nprocs = omp_get_num_procs();
    if (nprocs > 1) {
        return nprocs > CCUBES_THREAD_LIMIT ? CCUBES_THREAD_LIMIT : nprocs;
    }
#elif defined(HAVE_PTHREAD)
    #if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwNumberOfProcessors > 1) {
        return info.dwNumberOfProcessors > CCUBES_THREAD_LIMIT ?
            CCUBES_THREAD_LIMIT : (int)info.dwNumberOfProcessors;
    }
    #else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 1) {
        return nprocs > CCUBES_THREAD_LIMIT ? CCUBES_THREAD_LIMIT : (int)nprocs;
    }
    #endif
#endif
    return 1;
}

int ccubes_effective_thread_count(int requested_threads) {
    int nthreads = requested_threads;

    if (nthreads <= 0) {
        nthreads = ccubes_configured_threads > 0 ?
            ccubes_configured_threads : ccubes_default_thread_count();
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
    if (nthreads > CCUBES_THREAD_LIMIT) {
        nthreads = CCUBES_THREAD_LIMIT;
    }

#if !defined(HAVE_OPENMP) && !defined(HAVE_PTHREAD)
    nthreads = 1;
#endif

    return nthreads;
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
        return 1;
    }

#if defined(HAVE_OPENMP)
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int workers = omp_get_num_threads();
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

    uint64_t chunk = count / (uint64_t)nthreads;
    uint64_t rem = count % (uint64_t)nthreads;
    uint64_t next = start;
    int started = 0;
    int ok = 1;

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

        if (pthread_create(&threads[i], NULL, ccubes_thread_main, &tasks[i]) != 0) {
            ok = 0;
            break;
        }
        started++;
    }

    for (int i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(tasks);
    return ok;
#else
    (void)strided;
    worker(start, end, 1, 0, 1, data);
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
