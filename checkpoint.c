/*
    Copyright (c) 2016–2025, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define CK_MAGIC "CCCHKv1"  // 7 chars + \0
#define CK_VERSION 4u

static int write_bytes(FILE *f, const void *buf, size_t len) {
    return fwrite(buf, 1, len, f) == len ? 0 : -1;
}

static int read_bytes(FILE *f, void *buf, size_t len) {
    return fread(buf, 1, len, f) == len ? 0 : -1;
}

static int write_int(FILE *f, int v) {
    return write_bytes(f, &v, sizeof(int));
}
static int read_int(FILE *f, int *v) {
    return read_bytes(f, v, sizeof(int));
}

static int write_u32(FILE *f, uint32_t v) {
    return write_bytes(f, &v, sizeof(uint32_t));
}

static int read_u32(FILE *f, uint32_t *v) {
    return read_bytes(f, v, sizeof(uint32_t));
}

static int write_double(FILE *f, double v) {
    return write_bytes(f, &v, sizeof(double));
}
static int read_double(FILE *f, double *v) {
    return read_bytes(f, v, sizeof(double));
}

static int write_u64(FILE *f, uint64_t v) {
    return write_bytes(f, &v, sizeof(uint64_t));
}
static int read_u64(FILE *f, uint64_t *v) {
    return read_bytes(f, v, sizeof(uint64_t));
}

static int write_str(FILE *f, const char *s) {
    int len = (s ? (int)strlen(s) : -1);
    if (write_int(f, len) < 0) return -1;
    if (len > 0) {
        if (write_bytes(f, s, (size_t)len) < 0) return -1;
    }
    return 0;
}

static int read_str(FILE *f, char **out) {
    int len = 0;
    if (read_int(f, &len) < 0) return -1;
    if (len == -1) { *out = NULL; return 0; }
    if (len < 0) return -1;
    char *buf = (char*)calloc((size_t)len + 1u, 1u);
    if (!buf) return -1;
    if (read_bytes(f, buf, (size_t)len) < 0) { free(buf); return -1; }
    buf[len] = '\0';
    *out = buf;
    return 0;
}

int save_checkpoint(
    const char *path,
    PIstorage *PInfo,
    int ninputs,
    int noutputs,
    int bits_per_word,
    int value_bit_width,
    int implicant_words,
    int current_k,
    const int *stop_counter,
    int max_levels,
    int weight_pic,
    int scp_type,
    int pool_max,
    int start_level,
    const char *src_path,
    const char *dst_path,
    double elapsed_total,
    double elapsed_scp,
    uint64_t last_task
) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    // header
    char magic[8] = CK_MAGIC;
    if (write_bytes(f, magic, sizeof(magic)) < 0) goto FAIL;
    if (write_u32(f, CK_VERSION) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)ninputs) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)noutputs) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)bits_per_word) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)value_bit_width) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)implicant_words) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)current_k) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)max_levels) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)weight_pic) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)scp_type) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)pool_max) < 0) goto FAIL;
    if (write_u32(f, (uint32_t)start_level) < 0) goto FAIL;

    // src and dst paths (may be NULL)
    if (write_str(f, src_path) < 0) goto FAIL;
    if (write_str(f, dst_path) < 0) goto FAIL;

    // timing
    if (write_double(f, elapsed_total) < 0) goto FAIL;
    if (write_double(f, elapsed_scp) < 0) goto FAIL;

    // last_task at current_k
    if (write_u64(f, last_task) < 0) goto FAIL;

    // stop_counter per output
    for (int o = 0; o < noutputs; ++o) {
        if (write_int(f, stop_counter ? stop_counter[o] : 0) < 0) goto FAIL;
    }

    // For each output, write the necessary fields
    for (int o = 0; o < noutputs; ++o) {
        PIstorage *pi = &PInfo[o];
        if (write_int(f, pi->ON_minterms) < 0) goto FAIL;
        if (write_int(f, pi->OFF_minterms) < 0) goto FAIL;
        if (write_int(f, pi->foundPI) < 0) goto FAIL;
        if (write_int(f, pi->solmin) < 0) goto FAIL;
        if (write_int(f, pi->prevsolmin) < 0) goto FAIL;
        int flags = (pi->stop_search ? 1 : 0) | (pi->ON_set_covered ? 2 : 0);
        if (write_int(f, flags) < 0) goto FAIL;

        // nofpi[0..ninputs-1]
        if (write_bytes(f, pi->nofpi, (size_t)ninputs * sizeof(int)) < 0) goto FAIL;

        // ON_set and OFF_set data
        size_t on_sz = (size_t)pi->ON_minterms * (size_t)ninputs;
        size_t off_sz = (size_t)pi->OFF_minterms * (size_t)ninputs;
        if (write_bytes(f, pi->ON_set, on_sz * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->OFF_set, off_sz * sizeof(int)) < 0) goto FAIL;

        // search state arrays up to foundPI
        int pichart_words = pi->pichart_words;
        size_t fp = (size_t)pi->foundPI;
        size_t onm = (size_t)pi->ON_minterms;
        size_t ipw = (size_t)implicant_words;
        size_t pcw = (size_t)pichart_words;

        if (write_bytes(f, pi->covered, fp * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->last_index, onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->k_last_index, onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->pichart, fp * onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->pichart_pos, fp * pcw * sizeof(uint64_t)) < 0) goto FAIL;
        if (write_bytes(f, pi->implicants_pos, fp * ipw * sizeof(uint64_t)) < 0) goto FAIL;
        if (write_bytes(f, pi->implicants_val, fp * ipw * sizeof(uint64_t)) < 0) goto FAIL;
        if (write_bytes(f, pi->shared, fp * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->covsum, fp * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->previndices, onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->indices, onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->cov_word_index, onm * sizeof(int)) < 0) goto FAIL;
        if (write_bytes(f, pi->shifted_cov_mask, onm * sizeof(uint64_t)) < 0) goto FAIL;
    }

    fclose(f);
    return 0;
FAIL:
    fclose(f);
    return -1;
}

int load_checkpoint(
    const char *path,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int *bits_per_word,
    int *value_bit_width,
    int *implicant_words,
    int *current_k,
    int **stop_counter_out,
    int *max_levels,
    int *weight_pic,
    int *scp_type,
    int *pool_max,
    int *start_level,
    int **nofvalues_out,
    char **src_path_out,
    char **dst_path_out,
    double *elapsed_total_out,
    double *elapsed_scp_out,
    uint64_t *last_task_out
) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[8] = {0};
    if (read_bytes(f, magic, sizeof(magic)) < 0) goto FAIL;
    if (memcmp(magic, CK_MAGIC, sizeof(magic)) != 0) goto FAIL;

    uint32_t ver=0, ni=0, no=0, bpw=0, vbw=0, ipw=0, ck=0, ml=0, wp=0, st=0, pm=0, sl=0;
    if (read_u32(f, &ver) < 0) goto FAIL;

    if (read_u32(f, &ni) < 0) goto FAIL;
    if (read_u32(f, &no) < 0) goto FAIL;
    if (read_u32(f, &bpw) < 0) goto FAIL;
    if (read_u32(f, &vbw) < 0) goto FAIL;
    if (read_u32(f, &ipw) < 0) goto FAIL;
    if (read_u32(f, &ck) < 0) goto FAIL;
    if (read_u32(f, &ml) < 0) goto FAIL;
    if (read_u32(f, &wp) < 0) goto FAIL;
    if (read_u32(f, &st) < 0) goto FAIL;
    if (read_u32(f, &pm) < 0) goto FAIL;
    if (read_u32(f, &sl) < 0) goto FAIL;

    *ninputs = (int)ni;
    *noutputs = (int)no;
    *bits_per_word = (int)bpw;
    *value_bit_width = (int)vbw;
    *implicant_words = (int)ipw;
    *current_k = (int)ck;
    *max_levels = (int)ml;
    *weight_pic = (int)wp;
    *scp_type = (int)st;
    *pool_max = (int)pm;
    *start_level = (int)sl;

    // src and dst paths for version >= 2
    char *loaded_src = NULL, *loaded_dst = NULL;
    if (ver >= 2u) {
        if (read_str(f, &loaded_src) < 0) goto FAIL;
        if (read_str(f, &loaded_dst) < 0) { free(loaded_src); goto FAIL; }
    }
    double loaded_elapsed_total = 0.0, loaded_elapsed_scp = 0.0;
    uint64_t loaded_last_task = 0ull;
    if (ver >= 3u) {
        if (read_double(f, &loaded_elapsed_total) < 0) { free(loaded_src); free(loaded_dst); goto FAIL; }
        if (read_double(f, &loaded_elapsed_scp) < 0) { free(loaded_src); free(loaded_dst); goto FAIL; }
    }
    if (ver >= 4u) {
        if (read_u64(f, &loaded_last_task) < 0) { free(loaded_src); free(loaded_dst); goto FAIL; }
    }

    int *stopc = (int*)calloc((size_t)no, sizeof(int));
    if (!stopc) { free(loaded_src); free(loaded_dst); goto FAIL; }
    for (int o = 0; o < (int)no; ++o) {
        if (read_int(f, &stopc[o]) < 0) { free(stopc); free(loaded_src); free(loaded_dst); goto FAIL; }
    }
    *stop_counter_out = stopc;

    PIstorage *pi = (PIstorage*)calloc((size_t)no, sizeof(PIstorage));
    if (!pi) { free(loaded_src); free(loaded_dst); goto FAIL; }

    // initialize pointers to NULL for safe cleanup()
    for (int o = 0; o < (int)no; ++o) {
        pi[o].ON_set = NULL;
        pi[o].OFF_set = NULL;
        pi[o].covered = NULL;
        pi[o].last_index = NULL;
        pi[o].k_last_index = NULL;
        pi[o].pichart = NULL;
        pi[o].pichart_pos = NULL;
        pi[o].implicants_pos = NULL;
        pi[o].implicants_val = NULL;
        pi[o].shared = NULL;
        pi[o].covsum = NULL;
        pi[o].previndices = NULL;
        pi[o].indices = NULL;
        pi[o].cov_word_index = NULL;
        pi[o].shifted_cov_mask = NULL;
        pi[o].nofpi = NULL;
        pi[o].solution = NULL;
        pi[o].pool_solutions = NULL;
        pi[o].pool_count = 0;
    }

    for (int o = 0; o < (int)no; ++o) {
        PIstorage *po = &pi[o];
        if (read_int(f, &po->ON_minterms) < 0) goto READ_FAIL;
        if (read_int(f, &po->OFF_minterms) < 0) goto READ_FAIL;
        if (read_int(f, &po->foundPI) < 0) goto READ_FAIL;
        if (read_int(f, &po->solmin) < 0) goto READ_FAIL;
        if (read_int(f, &po->prevsolmin) < 0) goto READ_FAIL;
        int flags=0; if (read_int(f, &flags) < 0) goto READ_FAIL;
        po->stop_search = (flags & 1) != 0;
        po->ON_set_covered = (flags & 2) != 0;
        po->inputs = (int)ni;
        po->outputs = (int)no;
        po->pichart_words = (po->ON_minterms + (*bits_per_word) - 1) / (*bits_per_word);

        po->nofpi = (int*)calloc((size_t)ni, sizeof(int));
        if (!po->nofpi) goto READ_FAIL;
        if (read_bytes(f, po->nofpi, (size_t)ni * sizeof(int)) < 0) goto READ_FAIL;

        size_t on_sz = (size_t)po->ON_minterms * (size_t)ni;
        size_t off_sz = (size_t)po->OFF_minterms * (size_t)ni;
        po->ON_set = (int*)calloc(on_sz, sizeof(int));
        po->OFF_set = (int*)calloc(off_sz, sizeof(int));
        if ((on_sz && !po->ON_set) || (off_sz && !po->OFF_set)) goto READ_FAIL;
        if (read_bytes(f, po->ON_set, on_sz * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->OFF_set, off_sz * sizeof(int)) < 0) goto READ_FAIL;

        size_t fp = (size_t)po->foundPI;
        size_t onm = (size_t)po->ON_minterms;
        size_t ipw_s = (size_t)(*implicant_words);
        size_t pcw = (size_t)po->pichart_words;

        po->covered = (int*)calloc(fp, sizeof(int));
        po->last_index = (int*)calloc(onm, sizeof(int));
        po->k_last_index = (int*)calloc(onm, sizeof(int));
        po->pichart = (int*)calloc(fp * onm, sizeof(int));
        po->pichart_pos = (uint64_t*)calloc(fp * pcw, sizeof(uint64_t));
        po->implicants_pos = (uint64_t*)calloc(fp * ipw_s, sizeof(uint64_t));
        po->implicants_val = (uint64_t*)calloc(fp * ipw_s, sizeof(uint64_t));
        po->shared = (int*)calloc(fp, sizeof(int));
        po->covsum = (int*)calloc(fp, sizeof(int));
        po->previndices = (int*)calloc(onm, sizeof(int));
        po->indices = (int*)calloc(onm, sizeof(int));
        po->cov_word_index = (int*)calloc(onm, sizeof(int));
        po->shifted_cov_mask = (uint64_t*)calloc(onm, sizeof(uint64_t));

        if (
            (
                fp &&
                (
                    !po->covered ||
                    !po->pichart ||
                    !po->pichart_pos ||
                    !po->implicants_pos ||
                    !po->implicants_val ||
                    !po->shared ||
                    !po->covsum
                )
            ) ||
            !po->last_index ||
            !po->k_last_index ||
            !po->previndices ||
            !po->indices ||
            !po->cov_word_index ||
            !po->shifted_cov_mask
        ) goto READ_FAIL;

        if (read_bytes(f, po->covered, fp * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->last_index, onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->k_last_index, onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->pichart, fp * onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->pichart_pos, fp * pcw * sizeof(uint64_t)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->implicants_pos, fp * ipw_s * sizeof(uint64_t)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->implicants_val, fp * ipw_s * sizeof(uint64_t)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->shared, fp * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->covsum, fp * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->previndices, onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->indices, onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->cov_word_index, onm * sizeof(int)) < 0) goto READ_FAIL;
        if (read_bytes(f, po->shifted_cov_mask, onm * sizeof(uint64_t)) < 0) goto READ_FAIL;
    }

    fclose(f);

    // Build nofvalues from ON/OFF sets
    int *nofvals = (int*)calloc((size_t)(*ninputs), sizeof(int));
    if (!nofvals) {
        free(pi);
        free(*stop_counter_out);
        free(loaded_src);
        free(loaded_dst);
        return -1;
    }

    for (int j = 0; j < *ninputs; ++j) nofvals[j] = 0;

    for (int o = 0; o < *noutputs; ++o) {
        PIstorage *po = &pi[o];
        for (int r = 0; r < po->ON_minterms; ++r) {
            for (int j = 0; j < *ninputs; ++j) {
                int v = po->ON_set[r * (*ninputs) + j];
                if (v + 1 > nofvals[j]) nofvals[j] = v + 1;
            }
        }
        for (int r = 0; r < po->OFF_minterms; ++r) {
            for (int j = 0; j < *ninputs; ++j) {
                int v = po->OFF_set[r * (*ninputs) + j];
                if (v + 1 > nofvals[j]) nofvals[j] = v + 1;
            }
        }
    }

    *nofvalues_out = nofvals;

    if (src_path_out) {
        *src_path_out = loaded_src;
    } else {
        free(loaded_src);
    }

    if (dst_path_out) {
        *dst_path_out = loaded_dst;
    } else {
        free(loaded_dst);
    }

    if (elapsed_total_out) *elapsed_total_out = loaded_elapsed_total;
    if (elapsed_scp_out) *elapsed_scp_out = loaded_elapsed_scp;
    if (last_task_out) *last_task_out = loaded_last_task;

    *PInfo = pi;
    return 0;

READ_FAIL:
    // cleanup allocations on error
    for (int o = 0; o < (int)no; ++o) {
        free(pi[o].ON_set);
        free(pi[o].OFF_set);
        free(pi[o].covered);
        free(pi[o].last_index);
        free(pi[o].k_last_index);
        free(pi[o].pichart);
        free(pi[o].pichart_pos);
        free(pi[o].implicants_pos);
        free(pi[o].implicants_val);
        free(pi[o].shared);
        free(pi[o].covsum);
        free(pi[o].previndices);
        free(pi[o].indices);
        free(pi[o].cov_word_index);
        free(pi[o].shifted_cov_mask);
        free(pi[o].nofpi);
    }
    free(pi);
FAIL:
    fclose(f);
    return -1;
}

