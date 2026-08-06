#!/bin/sh
set -eu

ccubes=${1:-./ccubes}
fixture=${2:-examples/pool_positive.pla}
work=$(mktemp -d "${TMPDIR:-/tmp}/ccubes_pool_policy.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM

"$ccubes" -d -w2 -t1 -e0 "$fixture" "$work/w2.pla" \
    >"$work/w2.log" 2>&1
"$ccubes" -d -w1 -t1 -e0 -p -dbg1 "$fixture" "$work/pool.pla" \
    >"$work/pool.log" 2>&1

# The first feasible boundary improves the primary cover and must not enumerate.
grep -q "CCUBES_POOL_DEFER level=2 output=1 reason=primary-improved" \
    "$work/pool.log"
if grep -q "CCUBES_POOL_BUDGET level=2" "$work/pool.log"; then
    echo "pool alternatives were enumerated before the plateau boundary" >&2
    exit 1
fi

# The tied boundary is the only place where alternatives are collected.
grep -q \
    "CCUBES_POOL_BUDGET level=3 output=1 .* phase=plateau" \
    "$work/pool.log"
grep -q \
    "CCUBES_POOL level=3 .* connections=6 selected_rows=3 .* savings=3" \
    "$work/pool.log"

metrics() {
    awk '
        !/^[.#]/ && NF >= 2 {
            rows++
            for (i = 1; i <= length($2); ++i) {
                if (substr($2, i, 1) == "1") connections++
            }
        }
        END { print rows + 0, connections + 0 }
    ' "$1"
}

set -- $(metrics "$work/w2.pla")
w2_rows=$1
w2_connections=$2
set -- $(metrics "$work/pool.pla")
pool_rows=$1
pool_connections=$2

if [ "$pool_rows" -gt "$w2_rows" ]; then
    echo "pooling worsened the protected product-row incumbent" >&2
    exit 1
fi
if [ "$pool_connections" -gt "$w2_connections" ]; then
    echo "pooling worsened the protected per-output cardinality" >&2
    exit 1
fi

echo "pool boundary policy regression: OK"
