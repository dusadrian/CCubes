#!/bin/sh
set -eu

ccubes=${1:-./ccubes}
fixture=${2:-examples/certified_F2.pla}
work=$(mktemp -d "${TMPDIR:-/tmp}/ccubes_effort_policy.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

"$ccubes" -d -s0 -e0 -g -dbg1 "$fixture" "$work/e0.pla" \
    >"$work/e0.log" 2>&1
grep -q "hybrid effort 0 uses fast heuristic plateau stopping" "$work/e0.log"
grep -q \
    "CCUBES_PLATEAU_PROBE level=2 output=1 .* appended=1 improved=yes" \
    "$work/e0.log"
if grep -q "action=certify" "$work/e0.log"; then
    echo "-e0 unexpectedly triggered automatic certification" >&2
    exit 1
fi
if grep -q "^k: 3$" "$work/e0.log"; then
    echo "-e0 did not use the plateau probe's earlier one-term cover" >&2
    exit 1
fi

"$ccubes" -d -s0 -e0 -p -dbg1 "$fixture" "$work/e0_pool.pla" \
    >"$work/e0_pool.log" 2>&1
grep -q \
    "CCUBES_PLATEAU_PROBE level=2 output=1 .* appended=1 improved=yes pool" \
    "$work/e0_pool.log"
if grep -q "^k: 3$" "$work/e0_pool.log"; then
    echo "-e0 pooled execution did not use the plateau probe" >&2
    exit 1
fi

"$ccubes" -d -s0 -e2 -g -dbg1 "$fixture" "$work/e2.pla" \
    >"$work/e2.log" 2>&1
grep -q \
    "CCUBES_PLATEAU_PROBE level=2 output=1 .* appended=1 improved=yes" \
    "$work/e2.log"
if grep -q "^k: 3$" "$work/e2.log"; then
    echo "-e2 did not use the plateau probe's earlier one-term cover" >&2
    exit 1
fi
grep -q \
    "effort=2 certification_requested=0 iteration_limit=1000 portfolio_limit=1 polish_nodes=400000" \
    "$work/e2.log"
grep -q "warm_start_requested=1 warm_start_accepted=1" "$work/e2.log"

"$ccubes" -d -s0 -e0 -c -dbg1 "$fixture" "$work/e0_certified.pla" \
    >"$work/e0_certified.log" 2>&1
grep -q \
    "CCUBES_PI_GENERATOR level=1 selected=complete-mmcs" \
    "$work/e0_certified.log"
grep -q \
    "CCUBES_CERTIFICATE output=1 scope=global method=cardinality-floor status=certified level=1 cover=1" \
    "$work/e0_certified.log"

"$ccubes" -d -s0 -e2 -c -dbg1 "$fixture" "$work/e2_certified.pla" \
    >"$work/e2_certified.log" 2>&1
grep -q \
    "effort=2 certification_requested=1 iteration_limit=5000 portfolio_limit=6 polish_nodes=2000000" \
    "$work/e2_certified.log"
grep -q \
    "CCUBES_PI_GENERATOR level=1 selected=complete-mmcs" \
    "$work/e2_certified.log"
grep -q \
    "CCUBES_CERTIFICATE output=1 scope=global method=cardinality-floor status=certified level=1 cover=1" \
    "$work/e2_certified.log"

echo "hybrid effort stopping policy regression: OK"
