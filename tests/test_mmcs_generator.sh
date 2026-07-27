#!/bin/sh
set -eu

bin=${1:-./ccubes}
forced_fixture=${2:-examples/mmcs_forced_100x1.pla}
small_fixture=${3:-examples/certified_F2.pla}
dense_fixture=${4:-examples/rnd_20x10x40.pla}
tmp_prefix=/tmp/ccubes_mmcs_generator_$$
trap 'rm -f "${tmp_prefix}"_*' EXIT HUP INT TERM

"$bin" -t1 -d -e0 -dbg1 \
    "$forced_fixture" "${tmp_prefix}_forced.pla" \
    >"${tmp_prefix}_forced.log" 2>&1

grep -q '^000000---------------------------------------------------------------------------------------------- 1$' \
    "${tmp_prefix}_forced.pla"
grep -q 'CCUBES_PI_GENERATOR level=6 selected=mmcs' \
    "${tmp_prefix}_forced.log"

"$bin" -t1 -d -e0 --pi-generator=projection \
    "$small_fixture" "${tmp_prefix}_projection.pla" \
    >"${tmp_prefix}_projection.log" 2>&1
"$bin" -t1 -d -e0 --pi-generator=mmcs \
    "$small_fixture" "${tmp_prefix}_mmcs.pla" \
    >"${tmp_prefix}_mmcs.log" 2>&1

cmp "${tmp_prefix}_projection.pla" "${tmp_prefix}_mmcs.pla"

"$bin" -t4 -d -e0 -dbg1 \
    "$dense_fixture" "${tmp_prefix}_auto_dense.pla" \
    >"${tmp_prefix}_auto_dense.log" 2>&1
"$bin" -t4 -d -e0 --pi-generator=projection \
    "$dense_fixture" "${tmp_prefix}_projection_dense.pla" \
    >"${tmp_prefix}_projection_dense.log" 2>&1

grep -q 'selected=projection reason=mmcs-node-limit' \
    "${tmp_prefix}_auto_dense.log"
cmp "${tmp_prefix}_auto_dense.pla" "${tmp_prefix}_projection_dense.pla"
echo "bounded MMCS production integration: OK"
