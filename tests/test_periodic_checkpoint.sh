#!/bin/sh
set -eu

bin=${1:-./ccubes}
fixture=${2:-examples/rnd_20x10x40.pla}
tmp_prefix=/tmp/ccubes_periodic_checkpoint_$$
trap 'rm -f "${tmp_prefix}"_*' EXIT HUP INT TERM

"$bin" --threads=4 --deterministic --effort=0 \
    "$fixture" "${tmp_prefix}_baseline.pla" \
    >"${tmp_prefix}_baseline.log" 2>&1

"$bin" --threads=4 --deterministic --effort=0 --checkpoint-every=0.005 \
    --time-limit=0.025 --checkpoint="${tmp_prefix}_run.chk" \
    "$fixture" "${tmp_prefix}_resumed.pla" \
    >"${tmp_prefix}_checkpoint.log" 2>&1

grep -q 'Periodic checkpoint saved at .*; continuing\.' \
    "${tmp_prefix}_checkpoint.log"
grep -q 'Time limit reached\. Checkpoint saved at ' \
    "${tmp_prefix}_checkpoint.log"
test -s "${tmp_prefix}_run.chk"
test ! -e "${tmp_prefix}_run.chk.tmp"

"$bin" --inspect="${tmp_prefix}_run.chk" \
    >"${tmp_prefix}_info.log" 2>&1
grep -q 'Progress at k: resume_task=' "${tmp_prefix}_info.log"

"$bin" --threads=4 --deterministic --effort=0 --resume="${tmp_prefix}_run.chk" \
    >"${tmp_prefix}_resume.log" 2>&1

cmp "${tmp_prefix}_baseline.pla" "${tmp_prefix}_resumed.pla"

if "$bin" --checkpoint-every=invalid "$fixture" \
    >"${tmp_prefix}_invalid.log" 2>&1; then
    echo "invalid periodic checkpoint interval unexpectedly succeeded" >&2
    exit 1
fi
grep -q 'Invalid value for --checkpoint-every: invalid' \
    "${tmp_prefix}_invalid.log"

echo "periodic checkpoint integration: OK"
