#include "cli.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static CliOptions parsed;
static FILE *errors;

static int parse(char **args) {
    int count = 0;
    while (args[count]) ++count;
    return cli_parse(count, args, &parsed, errors);
}

#define PARSE(...) parse((char *[]){"ccubes", __VA_ARGS__, NULL})

int main(void) {
    errors = tmpfile();
    assert(errors);
    assert(PARSE("input.pla") == 0);
    assert(parsed.bits_per_word == 64 && parsed.threads == 0);
    assert(parsed.weights == 1 && parsed.solver == 0 && parsed.effort == 0);
    assert(!parsed.debug && !parsed.pool && !parsed.deterministic && !parsed.lock_stats);

    assert(PARSE("--bits-per-word", "128", "--threads=4", "--weights", "none",
                 "--solver=gurobi", "--effort", "2", "--deterministic",
                 "--certify", "--blocking-diagnostic", "--lock-stats",
                 "input.pla", "output.pla") == 0);
    assert(parsed.bits_per_word == 128 && parsed.threads == 4);
    assert(parsed.weights == 0 && parsed.solver == 1 && parsed.effort == 2);
    assert(parsed.deterministic && parsed.certify && parsed.blocking_diagnostic && parsed.lock_stats);
    assert(!strcmp(parsed.destination, "output.pla") && !parsed.legacy_options);
    assert(PARSE("--pool", "--weights=none", "input.pla") == 0);
    assert(parsed.pool && parsed.weights == 2);
    assert(PARSE("--weights=sharing", "--solver=hybrid", "--threads=0") == 0);
    assert(parsed.weights == 2 && parsed.solver == 0 && parsed.threads == 0);
    assert(PARSE("--time-limit", "12.5", "--checkpoint-every=.5",
                 "--checkpoint", "dir/a=b.chk", "--resume=old.chk",
                 "--debug-file", "dir/debug log", "--debug=2") == 0);
    assert(parsed.time_limit == 12.5 && parsed.checkpoint_every == .5);
    assert(!strcmp(parsed.checkpoint, "dir/a=b.chk") && !strcmp(parsed.resume, "old.chk"));
    assert(parsed.debug && parsed.debug_level == 2 && !strcmp(parsed.debug_file, "dir/debug log"));
    assert(PARSE("--inspect-level=1", "--inspect", "file.chk") == 0);
    assert(parsed.inspect_level == 1 && !strcmp(parsed.inspect, "file.chk"));
    assert(PARSE("--", "-source.pla", "--destination.pla") == 0);
    assert(!strcmp(parsed.source, "-source.pla") && !strcmp(parsed.destination, "--destination.pla"));

    /* Compatibility also exercises immutable string literals: the old parser
     * wrote into argv when separating combined level/path arguments. */
    assert(PARSE("-b32", "-t2", "-w2", "-s1", "-e", "1", "-d", "-c", "-g", "-p",
                 "-l12.5=dir/a=b.chk", "-r=old.chk", "-dbg2=debug.log") == 0);
    assert(parsed.legacy_options && parsed.bits_per_word == 32 && parsed.threads == 2);
    assert(parsed.weights == 2 && parsed.solver == 1 && parsed.effort == 1);
    assert(parsed.deterministic && parsed.certify && parsed.blocking_diagnostic && parsed.pool);
    assert(parsed.time_limit == 12.5 && !strcmp(parsed.checkpoint, "dir/a=b.chk"));
    assert(!strcmp(parsed.resume, "old.chk") && !strcmp(parsed.debug_file, "debug.log"));
    assert(parsed.debug && parsed.debug_level == 2);
    assert(PARSE("-i1=meta.chk", "-debug1", "-rold.chk") == 0);
    assert(parsed.inspect_level == 1 && !strcmp(parsed.inspect, "meta.chk"));
    assert(parsed.debug_level == 1 && !strcmp(parsed.resume, "old.chk"));
    assert(PARSE("-i=file.chk", "-dbg=log", "-r", "resume.chk") == 0);
    assert(parsed.inspect_level == 0 && parsed.debug_level == 0 && !strcmp(parsed.debug_file, "log"));
    assert(PARSE("-dbg") == 0 && parsed.debug && parsed.debug_level == 0);
    assert(PARSE("-h") == 1 && !parsed.legacy_options);
    assert(PARSE("--help") == 1);

    const char *invalid[] = {
        "--weights=garbage", "--weights=1x", "--solver=2", "--solver=", "--effort=3",
        "--threads=-1", "--threads=999999999999999999999999", "--threads=1.5",
        "--bits-per-word=7", "--bits-per-word=64junk", "--debug=3", "--debug=word",
        "--time-limit=nan", "--time-limit=inf", "--time-limit=-1", "--time-limit=1e9999",
        "--checkpoint-every=1x", "--checkpoint-every=", "--checkpoint-every= 1",
        "--pool=1", "--deterministic=false", "--lock-stats=1", "--resume=",
        "--inspect=", "--debug-file=", "--checkpoint=", "--threa=2", "--help=yes",
        "-p2", "-wgarbage", "-sxyz", "-b64x", "-dbgfoo", "-i2=meta.chk",
        "-i0=", "-i0", "-l1=", "-dbg1=", "-r=", "-dg", "--unknown"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        assert(PARSE((char *)invalid[i]) == -1);
    }
    assert(PARSE("--threads") == -1);
    assert(PARSE("--debug-file", "--certify") == -1);
    assert(PARSE("--checkpoint=run.chk") == -1);
    assert(PARSE("--inspect-level=1") == -1);
    assert(PARSE("one", "two", "three") == -1);

#ifdef CCUBES_TESTING
    assert(PARSE("--test-pi-generator=mmcs", "--test-projection-rwls") == 0);
    assert(parsed.test_pi_generator == PI_GENERATOR_MMCS && parsed.test_projection_rwls);
    assert(PARSE("--test-pi-generator=unknown") == -1);
#else
    assert(PARSE("--test-pi-generator=mmcs") == -1);
    assert(PARSE("--test-projection-rwls") == -1);
#endif
    fclose(errors);
    puts("CLI parsing regression: OK");
    return 0;
}
