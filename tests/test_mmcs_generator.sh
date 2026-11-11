#!/bin/sh
set -eu

bin=${1:-./ccubes}
forced_fixture=${2:-examples/mmcs_forced_100x1.pla}
small_fixture=${3:-examples/certified_F2.pla}
dense_fixture=${4:-examples/rnd_20x10x40.pla}
complete_fixture=${5:-examples/certified_two_term.pla}
test_bin=${6:-./.build/ccubes-testing}
tmp_prefix=/tmp/ccubes_mmcs_generator_$$
trap 'rm -f "${tmp_prefix}"_*' EXIT HUP INT TERM

"$bin" --threads=1 --effort=0 --debug=1 \
    "$forced_fixture" "${tmp_prefix}_forced.pla" \
    >"${tmp_prefix}_forced.log" 2>&1

grep -q '^000000---------------------------------------------------------------------------------------------- 1$' \
    "${tmp_prefix}_forced.pla"
grep -q 'CCUBES_PI_GENERATOR level=6 selected=mmcs' \
    "${tmp_prefix}_forced.log"
grep -q 'deterministic PI order: no' \
    "${tmp_prefix}_forced.log"

"$test_bin" --test-pi-generator=projection --threads=1 --deterministic --effort=0 --debug=1 \
    "$small_fixture" "${tmp_prefix}_projection.pla" \
    >"${tmp_prefix}_projection.log" 2>&1
"$test_bin" --test-pi-generator=mmcs --threads=1 --deterministic --effort=0 \
    "$small_fixture" "${tmp_prefix}_mmcs.pla" \
    >"${tmp_prefix}_mmcs.log" 2>&1

grep -q 'deterministic PI order: yes' \
    "${tmp_prefix}_projection.log"
cmp "${tmp_prefix}_projection.pla" "${tmp_prefix}_mmcs.pla"

if "$test_bin" --test-pi-generator=complete-mmcs --threads=1 \
    "$complete_fixture" "${tmp_prefix}_complete_without_c.pla" \
    >"${tmp_prefix}_complete_without_c.log" 2>&1; then
    echo "complete-mmcs unexpectedly ran without --certify" >&2
    exit 1
fi
grep -q 'complete MMCS is a global certification mode and requires --certify' \
    "${tmp_prefix}_complete_without_c.log"

if "$test_bin" --test-pi-generator=projection --threads=1 --certify \
    "$complete_fixture" "${tmp_prefix}_legacy_certificate.pla" \
    >"${tmp_prefix}_legacy_certificate.log" 2>&1; then
    echo "--certify unexpectedly accepted the retired horizon generator" >&2
    exit 1
fi
grep -q -- \
    '--certify uses complete-prime certification and cannot be combined with projection' \
    "${tmp_prefix}_legacy_certificate.log"

"$bin" --threads=1 --deterministic --certify --solver=hybrid --effort=2 --weights=none --debug=1 \
    "$complete_fixture" "${tmp_prefix}_complete.pla" \
    >"${tmp_prefix}_complete.log" 2>&1
grep -q 'Notice: --certify uses complete-prime MMCS certification' \
    "${tmp_prefix}_complete.log"
grep -q 'CCUBES_PI_GENERATOR level=1 selected=complete-mmcs' \
    "${tmp_prefix}_complete.log"
grep -q \
    'CCUBES_CERTIFICATE output=1 scope=global method=complete-prime-chart status=certified level=1 cover=2 boundary_exact=1' \
    "${tmp_prefix}_complete.log"
grep -q \
    'CCUBES_CERTIFICATION_ENUMERATION output=1 primes=2 nodes=' \
    "${tmp_prefix}_complete.log"
grep -q \
    'CCUBES_CERTIFICATION_CHART output=1 rows=2 primes=2 cover=2 boundary_exact=1 seconds=' \
    "${tmp_prefix}_complete.log"
grep -q \
    'CCUBES_CERTIFICATION_TIMING phase=summary .*outputs=1 primes=2' \
    "${tmp_prefix}_complete.log"
grep -q '^000 1$' "${tmp_prefix}_complete.pla"
grep -q '^111 1$' "${tmp_prefix}_complete.pla"

"$bin" --threads=4 --deterministic --effort=0 --debug=1 \
    "$dense_fixture" "${tmp_prefix}_auto_dense.pla" \
    >"${tmp_prefix}_auto_dense.log" 2>&1
"$test_bin" --test-pi-generator=projection --threads=4 --deterministic --effort=0 \
    "$dense_fixture" "${tmp_prefix}_projection_dense.pla" \
    >"${tmp_prefix}_projection_dense.log" 2>&1

grep -q 'selected=projection reason=mmcs-node-limit' \
    "${tmp_prefix}_auto_dense.log"
cmp "${tmp_prefix}_auto_dense.pla" "${tmp_prefix}_projection_dense.pla"
echo "bounded MMCS production integration: OK"
