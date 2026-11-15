#include "native_sat_verify.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NsatLimits unlimited = {0};
#ifdef NSAT_TESTING
static bool allocation_fails;
void *nsat_test_malloc(size_t bytes) { return allocation_fails ? NULL : malloc(bytes); }
#endif

/* A replayable test trace. O and S are trusted-CNF setup commands. Production
 * cover replay passes a separately initialized, already sealed verifier. */
static NsatStatus event(NsatVerifier *v, const char *line, bool trusted_cnf)
{
    size_t count, hint_count;
    NsatClauseId id, hints[16];
    int lits[16], used;
    switch (*line++) {
    case 'O':
        if (!trusted_cnf) return NSAT_ERROR;
        assert(sscanf(line, "%zu%n", &count, &used) == 1 && count <= 16);
        line += used;
        for (size_t i = 0; i < count; ++i) {
            assert(sscanf(line, "%d%n", &lits[i], &used) == 1);
            line += used;
        }
        return nsat_verify_add_original(v, lits, count);
    case 'S':
        return trusted_cnf ? nsat_verify_seal_originals(v) : NSAT_ERROR;
    case 'L':
        assert(sscanf(line, "%" SCNu64 " %zu%n", &id, &count, &used) == 2 && count <= 16);
        line += used;
        for (size_t i = 0; i < count; ++i) {
            assert(sscanf(line, "%d%n", &lits[i], &used) == 1);
            line += used;
        }
        assert(sscanf(line, "%zu%n", &hint_count, &used) == 1 && hint_count <= 16);
        line += used;
        for (size_t i = 0; i < hint_count; ++i) {
            assert(sscanf(line, "%" SCNu64 "%n", &hints[i], &used) == 1);
            line += used;
        }
        return nsat_verify_derive(v, id, lits, count, hints, hint_count);
    case 'D':
        assert(sscanf(line, "%" SCNu64, &id) == 1);
        return nsat_verify_delete(v, id);
    case 'C':
        assert(sscanf(line, "%" SCNu64, &id) == 1);
        return nsat_verify_conclude_unsat(v, id);
    default:
        abort();
    }
}

static NsatStatus replay(NsatVerifier *v, FILE *events, bool trusted_cnf)
{
    char line[512];
    NsatStatus status = NSAT_OK;
    while (status == NSAT_OK && fgets(line, sizeof(line), events))
        status = event(v, line, trusted_cnf);
    return status;
}

static NsatStatus trace(const char *events)
{
    NsatVerifier *v;
    assert(nsat_verify_init_cnf(&v, 3, unlimited) == NSAT_OK);
    FILE *file = tmpfile();
    assert(file && fputs(events, file) >= 0);
    rewind(file);
    NsatStatus result = replay(v, file, true);
    fclose(file);
    if (result == NSAT_ERROR) {
        NsatStopReason why = nsat_verify_stats(v).stop;
        assert(why != NSAT_STOP_NONE);
        assert(nsat_verify_conclude_unsat(v, 1) == NSAT_ERROR);
        assert(nsat_verify_stats(v).stop == why);
    }
    nsat_verify_destroy(v);
    return result;
}

static uint32_t random_word(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void test_sequences(void)
{
    /* Clause 2 implies x2 using x1, then clause 3 contradicts x2.
     * Variants exercise both proof and lifecycle failures. */
    const char *base = "O 1 1\nO 2 -1 2\nO 1 -2\nS\n";
    const char *suffixes[] = {
        "L 4 1 2 2 1 2\nL 5 0 2 4 3\nD 4\nC 5\n",
        "L 4 0 3 1 2 3\nC 4\n",
        "L 4 0 3 3 2 1\nC 4\n",
        "L 4 0 3 2 1 3\n",               /* first hint has two unassigned */
        "L 4 1 2 3 3 1 2\n",            /* early conflict */
        "L 4 1 2 3 1 1 2\n",            /* repeated hint already satisfied */
        "L 4 1 2 1 1\n",                /* no final conflict */
        "L 4 0 3 1 2 999\n",             /* unknown hint */
        "L 4 2 2 2 2 1 2\n",            /* duplicate literal */
        "L 4 2 2 -2 2 1 2\n",           /* complementary literals */
        "L 4 1 0 2 1 2\n",              /* invalid literal */
        "L 4 1 4 2 1 2\n",              /* out of range */
        "L 4 1 2 0\n",                  /* missing hints */
        "L 4 1 2 2 1 2\nC 4\n",         /* nonempty conclusion */
        "L 4 1 2 2 1 2\nD 4\nL 5 0 2 4 3\n",
        "L 4 1 2 2 1 2\nD 4\nL 4 1 2 2 1 2\n",
        "D 1\n",                        /* deletion of original */
        "C 2\n",                        /* claim without an empty clause */
        "S\n",
        "O 0\n"
    };
    size_t cases = sizeof(suffixes) / sizeof(*suffixes);
    for (size_t i = 0; i < cases; ++i) {
        char events[256];
        snprintf(events, sizeof(events), "%s%s", base, suffixes[i]);
        assert(trace(events) == (i < 3 ? NSAT_UNSAT : NSAT_ERROR));
    }
    const uint32_t seed = UINT32_C(0x7aba521);
    uint32_t rng = seed;
    for (size_t i = 0; i < 256; ++i) {
        int a = 1 + (int)(random_word(&rng) % 3);
        int b = a % 3 + 1;
        if (random_word(&rng) & 1) b = b % 3 + 1;
        if (random_word(&rng) & 1) a = -a;
        if (random_word(&rng) & 1) b = -b;
        NsatClauseId id = UINT64_C(4) + random_word(&rng);
        unsigned mode = random_word(&rng) % 4;
        char events[256];
        snprintf(events, sizeof(events),
                 "O 1 %d\nO 2 %d %d\nO 1 %d\nS\n"
                 "L %" PRIu64 " 1 %d 2 1 2\n%s"
                 "L %" PRIu64 " 0 2 %" PRIu64 " %u\nC %" PRIu64 "\n",
                 a, -a, b, -b, id, mode == 1 ? -b : b,
                 mode == 2 ? "D 1\n" : "",
                 id + 1, id, mode == 3 ? 2u : 3u, id + 1);
        NsatStatus actual = trace(events);
        NsatStatus expected = mode ? NSAT_ERROR : NSAT_UNSAT;
        if (actual != expected) {
            fprintf(stderr, "Replay failure seed=%" PRIu32 " case=%zu\n%s", seed, i, events);
            abort();
        }
    }
    assert(trace("L 1 0 1 1\n") == NSAT_ERROR);
    assert(trace("O 0\nS\nC 1\n") == NSAT_UNSAT);
    assert(trace("O 1 1\nS\nL 2 0 1 1\nC 2\n") == NSAT_ERROR);
    assert(trace("O 1 1\nS\n") == NSAT_OK);
    printf("Checker replay: 256 generated sequences and %zu fixed traces, seed=%" PRIu32 "\n",
           cases, seed);
}

static void test_cover_replay(void)
{
    const int on[] = {1, 1, 2, 2}, off[] = {1, 2};
    NsatVerifier *v;
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 1, NULL, 0, unlimited) == NSAT_OK);
    assert(event(v, "O 0", false) == NSAT_ERROR);
    FILE *file = tmpfile();
    assert(file && fputs("L 15 0 5 1 2 6 8 14\nC 15\n", file) >= 0);
    rewind(file);
    assert(replay(v, file, false) == NSAT_UNSAT);
    nsat_verify_destroy(v);
    /* The same trace does not certify the two-slot query. */
    rewind(file);
    assert(nsat_verify_init_cover(&v, on, off, 2, 2, 1, 2, NULL, 0, unlimited) == NSAT_OK);
    assert(replay(v, file, false) == NSAT_ERROR);
    fclose(file);
    nsat_verify_destroy(v);
}

static NsatVerifier *populate(size_t added)
{
    NsatVerifier *v;
    assert(nsat_verify_init_cnf(&v, 1, unlimited) == NSAT_OK);
    int lit = 1;
    NsatClauseId hint = 1;
    assert(nsat_verify_add_original(v, &lit, 1) == NSAT_OK);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    for (size_t i = 0; i < added; ++i)
        assert(nsat_verify_derive(v, UINT64_C(1) << 60 | i, &lit, 1, &hint, 1) == NSAT_OK);
    return v;
}

static void test_reclamation(void)
{
    NsatVerifier *v = populate(1024);
    size_t before = nsat_verify_stats(v).memory_bytes;
    for (size_t i = 0; i < 1024; ++i)
        assert(nsat_verify_delete(v, UINT64_C(1) << 60 | i) == NSAT_OK);
    size_t after = nsat_verify_stats(v).memory_bytes;
    assert(after < before / 10);
    NsatClauseId hint = 1;
    int lit = 1;
    for (NsatClauseId i = 0; i < 8192; ++i) {
        NsatClauseId id = UINT64_MAX - 8191 + i;
        assert(nsat_verify_derive(v, id, &lit, 1, &hint, 1) == NSAT_OK);
        assert(nsat_verify_delete(v, id) == NSAT_OK);
    }
    assert(nsat_verify_stats(v).memory_bytes == after);
    assert(nsat_verify_derive(v, 0, &lit, 1, &hint, 1) == NSAT_ERROR);
    nsat_verify_destroy(v);

#ifdef NSAT_TESTING
    v = populate(15); /* 32 entries, rebuild when the live count falls to 4. */
    for (size_t i = 0; i < 11; ++i)
        assert(nsat_verify_delete(v, UINT64_C(1) << 60 | i) == NSAT_OK);
    allocation_fails = true;
    assert(nsat_verify_delete(v, (UINT64_C(1) << 60) | 11) == NSAT_ERROR);
    assert(nsat_verify_stats(v).stop == NSAT_STOP_ALLOCATION);
    nsat_verify_destroy(v);
    allocation_fails = false;
#endif
}

static bool cancel_after_polls(void *context)
{
    unsigned *left = context;
    if (!*left) return true;
    --*left;
    return false;
}

static void test_cancellation_inside_proof(void)
{
    enum { WIDTH = 10000 };
    int *lits = malloc(WIDTH * sizeof(*lits));
    assert(lits);
    for (int i = 0; i < WIDTH; ++i) lits[i] = i + 1;
    NsatVerifier *v;
    unsigned left = UINT_MAX;
    NsatLimits limits = {.poll = cancel_after_polls, .context = &left};
    assert(nsat_verify_init_cnf(&v, WIDTH, limits) == NSAT_OK);
    assert(nsat_verify_add_original(v, lits, WIDTH) == NSAT_OK);
    assert(nsat_verify_seal_originals(v) == NSAT_OK);
    left = 2;
    uint64_t work = nsat_verify_stats(v).work;
    NsatClauseId hint = 1;
    assert(nsat_verify_derive(v, 2, lits, WIDTH, &hint, 1) == NSAT_UNKNOWN);
    assert(nsat_verify_stats(v).work > work && nsat_verify_stats(v).work < work + WIDTH);
    assert(nsat_verify_stats(v).stop == NSAT_STOP_CANCELLED);
    assert(nsat_verify_conclude_unsat(v, 2) == NSAT_UNKNOWN);
    nsat_verify_destroy(v);
    free(lits);
}

int main(int argc, char **argv)
{
    if (argc == 2) {
        NsatVerifier *v;
        assert(nsat_verify_init_cnf(&v, 3, unlimited) == NSAT_OK);
        FILE *file = fopen(argv[1], "r");
        assert(file);
        NsatStatus result = replay(v, file, true);
        printf("Replay result=%d stop=%d\n", result, nsat_verify_stats(v).stop);
        fclose(file);
        nsat_verify_destroy(v);
        return result == NSAT_UNSAT ? 0 : 1;
    }
    test_sequences();
    test_cover_replay();
    test_reclamation();
    test_cancellation_inside_proof();
    return 0;
}
