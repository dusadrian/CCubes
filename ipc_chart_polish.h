/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
#ifndef CCUBES_IPC_CHART_POLISH_H
#define CCUBES_IPC_CHART_POLISH_H

#include "native_sat_verify.h"
#include "pichart_view.h"

typedef struct IpcChartPolish IpcChartPolish;

typedef struct {
    const unsigned char *allowed; /* Reduced-cost core, in chart order. */
    const int *forced;
    int forced_len, target, max_row_degree;
    const int *coverage_counts;
    const double *reduced_costs, *tie_weights;
    uint64_t node_limit;
} IpcChartPolishQuery;

typedef struct {
    NsatStatus status;
    int terms;
    uint64_t nodes, work;
    size_t peak_bytes;
    unsigned workers;
} IpcChartPolishResult;

/* The chart is borrowed. The root basis and work allowance survive target
 * descent. A limit or unavailable workspace leaves the target undecided. */
IpcChartPolish *ipc_chart_polish_open(const PIChartView *chart, size_t memory_bytes,
                                    uint64_t deadline_nanoseconds, unsigned workers);
IpcChartPolishResult ipc_chart_polish_decide(IpcChartPolish *polish,
    const IpcChartPolishQuery *query, int *solution);
void ipc_chart_polish_close(IpcChartPolish *polish);

#endif
