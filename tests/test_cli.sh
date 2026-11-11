#!/bin/sh
set -eu

bin=${1:-./ccubes}
fixture=examples/certified_F2.pla
work=$(mktemp -d "${TMPDIR:-/tmp}/ccubes_cli.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

"$bin" --help >"$work/help"
grep -q -- '--threads VALUE' "$work/help"
if grep -q -- '--test-' "$work/help"; then
    echo "test controls leaked into production help" >&2
    exit 1
fi

"$bin" --threads 1 --deterministic --solver hybrid --weights complexity --effort 0 \
    --debug 1 --debug-file "$work/debug.log" "$fixture" "$work/long.pla" \
    >"$work/long.log" 2>&1
grep -q 'deterministic PI order: yes' "$work/debug.log"
"$bin" -t1 -d -s0 -w1 -e0 -dbg1 "$fixture" "$work/short.pla" \
    >"$work/short.log" 2>&1
cmp "$work/long.pla" "$work/short.pla"
grep -q 'short options are deprecated' "$work/short.log"

# Explicit word width choices preserve the same cover.
for width in 8 16 32 64 128; do
    "$bin" --bits-per-word="$width" --threads=1 --deterministic \
        "$fixture" "$work/width.pla" >"$work/width.log" 2>&1
    cmp "$work/long.pla" "$work/width.pla"
done

# Stale experiment environments must have no effect, even in diagnostic output.
env CCUBES_DETERMINISTIC=1 CCUBES_LOCK_STATS=1 \
    CCUBES_TEST_PI_GENERATOR=invalid CCUBES_TEST_PROJECTION_RWLS=1 \
    "$bin" --threads=1 --debug=1 "$fixture" "$work/env.pla" \
    >"$work/env.log" 2>&1
cmp "$work/long.pla" "$work/env.pla"
grep -q 'deterministic PI order: no' "$work/env.log"
if grep -Eq 'lock contention|CCUBES_PROJECTION_RWLS' "$work/env.log"; then
    echo "retired environment settings still affect production" >&2
    exit 1
fi
"$bin" --threads=1 --lock-stats "$fixture" "$work/locks.pla" >"$work/locks.log" 2>&1
grep -q 'CCUBES lock contention' "$work/locks.log"

# Zero and omission select the same effective backend thread count.
"$bin" --threads=0 --debug=1 "$fixture" "$work/zero.pla" >"$work/zero.log" 2>&1
"$bin" --debug=1 "$fixture" "$work/auto.pla" >"$work/auto.log" 2>&1
grep '^threads: ' "$work/zero.log" >"$work/zero.threads"
grep '^threads: ' "$work/auto.log" >"$work/auto.threads"
cmp "$work/zero.threads" "$work/auto.threads"

# Invalid options fail before opening/truncating requested output or log files.
printf 'preserve\n' >"$work/protected.log"
for invalid in --test-projection-rwls --test-pi-generator=mmcs --weights=typo --threads=2junk; do
    if "$bin" --debug-file "$work/protected.log" "$invalid" "$fixture" "$work/invalid.pla" \
        >"$work/invalid.log" 2>&1; then
        echo "invalid production option accepted: $invalid" >&2
        exit 1
    fi
    test ! -e "$work/invalid.pla"
    grep -qx preserve "$work/protected.log"
done

echo "CLI and environment isolation integration: OK"
