#!/bin/sh
set -eu

ccubes=${1:-./ccubes}
exact_fixture=${2:-examples/certified_F2.pla}
improvement_fixture=${3:-examples/boom_15x1x300.pla}
test_bin=${4:-./.build/ccubes-testing}
work=$(mktemp -d "${TMPDIR:-/tmp}/ccubes_projection_rwls.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

"$ccubes" --threads=1 --deterministic --effort=0 "$exact_fixture" "$work/exact-base.pla" \
    >"$work/exact-base.log" 2>&1
"$test_bin" --test-projection-rwls --threads=1 --deterministic --effort=0 "$exact_fixture" "$work/exact-rwls.pla" \
    >"$work/exact-rwls.log" 2>&1

grep -q \
    'CCUBES_PROJECTION_RWLS level=2 output=1 status=skipped reason=boundary-exact' \
    "$work/exact-rwls.log"
cmp "$work/exact-base.pla" "$work/exact-rwls.pla"

"$ccubes" --threads=4 --deterministic --effort=0 \
    "$improvement_fixture" "$work/improvement-base.pla" \
    >"$work/improvement-base.log" 2>&1
"$test_bin" --test-projection-rwls --threads=4 --deterministic --effort=0 \
    "$improvement_fixture" "$work/improvement-rwls.pla" \
    >"$work/improvement-rwls.log" 2>&1

grep -q '^.p 27$' "$work/improvement-base.pla"
grep -q '^.p 25$' "$work/improvement-rwls.pla"
grep -q \
    'CCUBES_PROJECTION_RWLS level=7 output=1 status=improved cover=27->25' \
    "$work/improvement-rwls.log"

if command -v espresso >/dev/null 2>&1; then
    espresso -Dverify \
        "$improvement_fixture" "$work/improvement-rwls.pla" \
        >"$work/verify.log" 2>&1
    grep -q "PLA's compared equal" "$work/verify.log"
fi

echo "projection RWLS integration: OK"
