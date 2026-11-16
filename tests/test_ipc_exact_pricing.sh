#!/bin/sh
set -eu

bin=${1:-./ccubes}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
source=examples/certified_two_term.pla

"$bin" --help >"$work/help"
grep -q '^  --ipc-exact-pricing VALUE$' "$work/help"

"$bin" --route ipc --threads 1 "$source" "$work/default.pla" >"$work/default.log" 2>&1
grep -Eq '^CCUBES_PROGRESSIVE_NATIVE_EXACT_PRICE round=1 kind=global status=0 ' "$work/default.log"
grep -qx '\.p 2' "$work/default.pla"

"$bin" --route ipc --threads 1 --ipc-exact-pricing 0 "$source" "$work/off.pla" >"$work/off.log" 2>&1
if grep -q '^CCUBES_PROGRESSIVE_NATIVE_EXACT_PRICE' "$work/off.log"; then
    echo "exact pricing ran after being disabled" >&2
    exit 1
fi
cmp "$work/default.pla" "$work/off.pla"

if "$bin" --route ipc --ipc-exact-pricing 2 "$source" "$work/invalid.pla" >"$work/invalid.log" 2>&1; then
    echo "exact pricing accepted a value outside 0 and 1" >&2
    exit 1
fi
grep -q 'Invalid value for --ipc-exact-pricing' "$work/invalid.log"
test ! -e "$work/invalid.pla"

echo "IPC exact pricing CLI: OK"
