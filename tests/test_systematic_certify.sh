#!/bin/sh
set -eu

bin=${1:-./ccubes}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

"$bin" --certify-systematic --systematic-sat-seconds=0 --threads=1 \
    examples/certified_two_term.pla "$work/bound-cover.pla" >"$work/bound.log" 2>&1
grep -q 'cover=2 status=certified method=ipc-global-bound' "$work/bound.log"
grep -qx '\.p 2' "$work/bound-cover.pla"

# The weighted stage improves the three-term incumbent to two.
cat >"$work/sat.pla" <<'PLA'
.i 8
.o 1
.type fr
.p 10
01100000 1
10100000 1
11000000 1
11100000 1
00011000 1
00010000 0
00001000 0
00000100 0
00000010 0
00000001 0
.e
PLA
printf 'preserve\n' >"$work/unproved.pla"
if "$bin" --certify-systematic --systematic-sat-seconds=0 --threads=1 --deterministic \
    "$work/sat.pla" "$work/unproved.pla" >"$work/unproved.log" 2>&1; then
    echo "a bound with an open gap was accepted" >&2
    exit 1
fi
grep -q 'cover=3 status=unproved' "$work/unproved.log"
grep -qx preserve "$work/unproved.pla"

# Run weighted certification with SAT disabled and no external executables.
env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
    --systematic-native-sat-seconds=0 \
    "$work/sat.pla" "$work/sat-cover.pla" >"$work/sat.log" 2>&1
grep -q 'CCUBES_SYSTEMATIC_INTEGER output=1 status=certified .*incumbent=3 cover=2 lower=2 .*check_status=not-needed' "$work/sat.log"
grep -q 'cover=2 status=certified method=ipc-global-bound' "$work/sat.log"
test "$(grep -c '^CCUBES_SYSTEMATIC_SAT ' "$work/sat.log" || true)" -eq 0
grep -qx -- '---00000 1' "$work/sat-cover.pla"
grep -qx -- '---11--- 1' "$work/sat-cover.pla"

# If weighted search is skipped or times out, SAT can use its own full budget.
for weighted_seconds in 0 0.000000000001; do
    env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
        --systematic-weighted-seconds="$weighted_seconds" --systematic-native-sat-seconds=2 \
        "$work/sat.pla" "$work/sat-only.pla" >"$work/sat-only.log" 2>&1
    grep -q '^CCUBES_SYSTEMATIC_SAT .*target=2 status=sat' "$work/sat-only.log"
    cmp "$work/sat-cover.pla" "$work/sat-only.pla"
    if [ "$weighted_seconds" = 0 ]; then
        test "$(grep -c '^CCUBES_SYSTEMATIC_INTEGER ' "$work/sat-only.log" || true)" -eq 0
    else
        grep -q '^CCUBES_SYSTEMATIC_INTEGER .*status=unknown stop_reason=deadline' "$work/sat-only.log"
    fi
done

if "$bin" --certify-systematic --threads=1 --deterministic \
    --systematic-weighted-seconds=0 --systematic-native-sat-seconds=0.000000000001 \
    "$work/sat.pla" "$work/unproved.pla" >"$work/sat-limit.log" 2>&1; then
    echo "systematic certification succeeded after the SAT deadline" >&2
    exit 1
fi
grep -q '^CCUBES_SYSTEMATIC_SAT .*status=unknown stop_reason=deadline' "$work/sat-limit.log"
grep -qx preserve "$work/unproved.pla"

env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
    --systematic-strengthening=combined --systematic-descent=reuse \
    "$work/sat.pla" "$work/strengthened.pla" \
    >"$work/strengthened.log" 2>&1
grep -q 'cover=2 status=certified method=ipc-global-bound' "$work/strengthened.log"
grep -qx -- '---00000 1' "$work/strengthened.pla"
grep -qx -- '---11--- 1' "$work/strengthened.pla"

# Put the decisive literals on both sides of multiple word boundaries.
python3 - "$work/sat.pla" "$work/wide.pla" <<'PY'
from pathlib import Path
import sys

active = (0, 5, 9, 63, 64, 510, 511, 512)
rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.startswith(("0", "1")):
        continue
    point, value = line.split()
    padded = ["0"] * 513
    for index, bit in zip(active, point, strict=True):
        padded[index] = bit
    rows.append("".join(padded) + " " + value)
Path(sys.argv[2]).write_text(".i 513\n.o 1\n.type fr\n.p 10\n" + "\n".join(rows) + "\n.e\n")
PY
env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
    "$work/wide.pla" "$work/wide-cover.pla" >"$work/wide.log" 2>&1
grep -q 'cover=2 status=certified method=ipc-global-bound' "$work/wide.log"
grep -qx '\.p 2' "$work/wide-cover.pla"

# Add free input bits to both copies to make compatible-set enumeration
# hit its work allotment. The care rows still require the same four cubes.
python3 - "$work/sat.pla" "$work/descent.pla" <<'PYTHON'
from pathlib import Path
import sys

source = Path(sys.argv[1])
care = [line.split() for line in source.read_text().splitlines() if line.startswith(("0", "1"))]
rows = []
for tail in range(8):
    for point, value in care:
        for tag in ("00", "11"):
            rows.append(f"{point}{tag}{tail:03b} {value}")
        if value == "1":
            for tag in ("01", "10"):
                rows.append(f"{point}{tag}{tail:03b} 0")
Path(sys.argv[2]).write_text(
    f".i 13\n.o 1\n.type fr\n.p {len(rows)}\n" + "\n".join(rows) + "\n.e\n"
)
PYTHON
for descent in fresh reuse; do
    env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
        --systematic-weighted-work=50000000 --systematic-native-sat-seconds=10 \
        --systematic-descent="$descent" \
        "$work/descent.pla" "$work/descent-$descent.pla" >"$work/descent-$descent.log" 2>&1
    grep -q '^CCUBES_SYSTEMATIC_INTEGER output=1 status=unknown stop_reason=work-limit' \
        "$work/descent-$descent.log"
    grep -Eq 'target=5 status=sat .*reused=no formula_slots=5' "$work/descent-$descent.log"
    grep -q 'cover=4 status=certified method=ipc-global-bound' "$work/descent-$descent.log"
    grep -qx '\.p 4' "$work/descent-$descent.pla"
done
grep -Eq 'target=4 status=sat .*reused=no formula_slots=4' "$work/descent-fresh.log"
grep -Eq 'target=4 status=sat .*reused=yes formula_slots=5' "$work/descent-reuse.log"

# An exhausted weighted budget with SAT disabled leaves the destination intact.
if "$bin" --certify-systematic --threads=1 --deterministic \
    --systematic-weighted-work=1 --systematic-native-sat-seconds=0 \
    "$work/sat.pla" "$work/unproved.pla" >"$work/weighted-limit.log" 2>&1; then
    echo "systematic certification succeeded with an exhausted weighted budget" >&2
    exit 1
fi
grep -q '^CCUBES_SYSTEMATIC_INTEGER .*status=unknown stop_reason=work-limit' "$work/weighted-limit.log"
test "$(grep -c '^CCUBES_SYSTEMATIC_SAT ' "$work/weighted-limit.log" || true)" -eq 0
grep -qx preserve "$work/unproved.pla"

env PATH=/nonexistent "$bin" --certify-systematic --threads=1 --deterministic \
    --systematic-certification-seconds=12 --systematic-certification-work=100000000 \
    examples/rnd_20x10x40.pla "$work/proved.pla" >"$work/proved.log" 2>&1
grep -Eq 'scope=scalar .*status=certified method=(ipc-weighted-integer|native-rup)' "$work/proved.log"
test "$(grep -c 'scope=scalar .*status=certified' "$work/proved.log")" -eq 10
env PATH=/nonexistent "$bin" --certify-systematic --threads=12 --deterministic \
    --systematic-certification-seconds=12 --systematic-certification-work=100000000 \
    examples/rnd_20x10x40.pla "$work/proved-parallel.pla" >"$work/proved-parallel.log" 2>&1
grep -Eq 'CCUBES_SYSTEMATIC_BATCH outputs=10 .*workers=([1-9]|10) pricing_threads=1 failed=0' \
    "$work/proved-parallel.log"
test "$(grep -c 'scope=scalar .*status=certified' "$work/proved-parallel.log")" -eq 10
for log in "$work/proved.log" "$work/proved-parallel.log"; do
    awk '/^CCUBES_SYSTEMATIC_CERTIFICATE / { print $2, $4 }' "$log" | sort >"$log.counts"
done
cmp "$work/proved.log.counts" "$work/proved-parallel.log.counts"
env PATH=/nonexistent "$bin" --certify-systematic --threads=12 --deterministic \
    examples/rnd_30x10x100.pla "$work/proved30.pla" >"$work/proved30.log" 2>&1
test "$(grep -c 'scope=scalar .*status=certified' "$work/proved30.log")" -eq 10
test "$(grep -c '^CCUBES_SYSTEMATIC_INTEGER .*check_status=verified' "$work/proved30.log")" -eq 9
for output in 5 7; do
    grep -q "^CCUBES_SYSTEMATIC_INTEGER output=$output status=certified .*incumbent=8 cover=7 lower=7" "$work/proved30.log"
done
grep -Eq '^CCUBES_SYSTEMATIC_INTEGER output=7 .*check_status=verified .*lower_target=6 repriced=[1-9][0-9]*' \
    "$work/proved30.log"
test "$(grep -c '^CCUBES_SYSTEMATIC_SAT ' "$work/proved30.log" || true)" -eq 0

python3 - "$work" <<'PY'
from pathlib import Path
import sys

def rows(path):
    return [line.split() for line in Path(path).read_text().splitlines()
            if line.startswith(("0", "1", "-"))]

work = Path(sys.argv[1])
for source, result, counts in [
    ("rnd_20x10x40", "proved", [5] + [4] * 9),
    ("rnd_20x10x40", "proved-parallel", [5] + [4] * 9),
    ("rnd_30x10x100", "proved30", [7, 6] + [7] * 8),
]:
    care = rows(Path("examples") / (source + ".pla"))
    cover = rows(work / (result + ".pla"))
    assert [sum(out[o] == "1" for _, out in cover) for o in range(10)] == counts
    for point, expected in care:
        actual = [False] * len(expected)
        for cube, outputs in cover:
            if all(c == "-" or c == x for c, x in zip(cube, point, strict=True)):
                actual = [hit or out == "1" for hit, out in zip(actual, outputs, strict=True)]
        assert actual == [out == "1" for out in expected]
PY

printf 'preserve parallel destination\n' >"$work/unproved-parallel.pla"
if "$bin" --certify-systematic --threads=12 --deterministic --systematic-certification-work=1 \
    examples/rnd_20x10x40.pla "$work/unproved-parallel.pla" >"$work/parallel-limit.log" 2>&1; then
    echo "parallel certification succeeded with an exhausted work limit" >&2
    exit 1
fi
grep -q 'status=unknown stop_reason=work-limit' "$work/parallel-limit.log"
test "$(grep -c '^CCUBES_SYSTEMATIC_CERTIFICATE ' "$work/parallel-limit.log")" -eq 10
grep -qx 'preserve parallel destination' "$work/unproved-parallel.pla"

for limit in --systematic-certification-work=2 --systematic-certification-seconds=0.000000000001; do
    if "$bin" --certify-systematic --threads=1 --deterministic "$limit" \
        "$work/sat.pla" "$work/unproved.pla" >"$work/limit.log" 2>&1; then
        echo "systematic certification succeeded with exhausted limit: $limit" >&2
        exit 1
    fi
    grep -q 'status=unproved' "$work/limit.log"
    grep -qx preserve "$work/unproved.pla"
    case "$limit" in
        *-work=*)
            grep -q '^CCUBES_SYSTEMATIC_INTEGER .*status=unknown stop_reason=work-limit' "$work/limit.log"
            grep -q '^CCUBES_SYSTEMATIC_SAT .*status=unknown stop_reason=work-limit' "$work/limit.log"
            ;;
        *-seconds=*) grep -q 'stop_reason=deadline' "$work/limit.log" ;;
    esac
done

for route in ipc auto; do
    if "$bin" --certify-systematic --route="$route" --threads=1 \
        examples/certified_two_term.pla "$work/route.pla" >"$work/route.log" 2>&1; then
        test "$route" = auto
        grep -q 'decision=e0 .*reason=systematic-certification' "$work/route.log"
    else
        test "$route" = ipc
        grep -q 'requires ordinary minimization' "$work/route.log"
    fi
done

if "$bin" --certify-systematic --threads=1 examples/heuristic_pattern.pla \
    "$work/pattern.pla" >"$work/pattern.log" 2>&1; then
    echo "systematic certification accepted input pattern rows" >&2
    exit 1
fi
grep -q 'requires nonconflicting binary point rows' "$work/pattern.log"

echo "Systematic certification CLI integration: OK"
