#!/usr/bin/env python3
"""Reduce merged PLA rows by reusing geometries from a validated cover."""

import argparse
import json
from pathlib import Path

from joint_cover import joint_recombine
from pla_helpers import (
    ExplicitPointValidator,
    Pla,
    read_pla,
    write_pla,
)


def union_coverage(columns: list[int], coverage: list[int]) -> int:
    covered = 0
    for column in columns:
        covered |= coverage[column]
    return covered


def recombine(
    source: Pla, cover: Pla, target_rows: int | None = None
) -> list[tuple[str, str]]:
    if (source.ninputs, source.noutputs) != (cover.ninputs, cover.noutputs):
        raise ValueError("care table and cover dimensions differ")
    if not source.rows or any(
        set(point + values) - {"0", "1"} for point, values in source.rows
    ):
        raise ValueError("expected a nonempty, explicit binary care table")

    patterns = [pattern for pattern, _ in cover.rows]
    if len(set(patterns)) != len(patterns):
        raise ValueError("expected unique geometries in the merged cover")
    owners = []
    selected = [[] for _ in range(cover.noutputs)]
    for column, (_, values) in enumerate(cover.rows):
        if set(values) - {"0", "1"} or "1" not in values:
            raise ValueError("expected binary cover masks with an asserted output")
        outputs = {output for output, value in enumerate(values) if value == "1"}
        owners.append(outputs)
        for output in outputs:
            selected[output].append(column)

    validator = ExplicitPointValidator(source)
    coverage = [validator.cube_coverage(pattern) for pattern in patterns]
    required = validator.on_bits
    for output, columns in enumerate(selected):
        if union_coverage(columns, coverage) != required[output]:
            raise ValueError(f"starting cover disagrees with care table on output {output + 1}")

    remaining_rows = len(patterns)
    while True:
        before_sweep = remaining_rows
        for output, columns in enumerate(selected):
            if target_rows is not None and remaining_rows <= target_rows:
                break
            on = required[output]
            if on == 0 or on == validator.all_rows:
                continue

            # Only active rows on other outputs can replace a row at zero marginal
            # cost. Filtering by the full care table excludes every OFF point.
            reusable = [
                column for column, covered in enumerate(coverage)
                if owners[column] and output not in owners[column]
                and covered and not (covered & ~on)
            ]
            reusable_coverage = union_coverage(reusable, coverage)
            for column in list(columns):
                if len(owners[column]) != 1:
                    continue
                retained = [other for other in columns if other != column]
                uncovered = on & ~union_coverage(retained, coverage)
                if uncovered & ~reusable_coverage:
                    continue
                candidates = [
                    other for other in reusable
                    if output not in owners[other] and coverage[other] & uncovered
                ]

                # Ties follow the original row order. Commit each replacement before
                # computing the rows exposed by the next removal.
                while uncovered:
                    best = max(
                        candidates, key=lambda other: (coverage[other] & uncovered).bit_count()
                    )
                    retained.append(best)
                    owners[best].add(output)
                    uncovered &= ~coverage[best]
                columns[:] = retained
                owners[column].remove(output)
                remaining_rows -= 1
                if target_rows is not None and remaining_rows <= target_rows:
                    break

        if remaining_rows == before_sweep or remaining_rows == target_rows:
            break

    if target_rows is not None and remaining_rows > target_rows:
        raise ValueError(f"post-pass reached {remaining_rows} rows, above target {target_rows}")
    for output, columns in enumerate(selected):
        if union_coverage(columns, coverage) != required[output]:
            raise ValueError(f"recombined cover disagrees with care table on output {output + 1}")
    return [
        (pattern, "".join("1" if output in outputs else "0" for output in range(cover.noutputs)))
        for pattern, outputs in zip(patterns, owners) if outputs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="explicit binary care-table PLA")
    parser.add_argument("--cover", type=Path, required=True, help="starting merged cover PLA")
    parser.add_argument(
        "--target-rows", type=int,
        help="stop at this merged row count (default: sweep until no more rows can be removed)",
    )
    parser.add_argument("--output", type=Path, required=True, help="destination PLA")
    parser.add_argument("--joint", action="store_true",
                        help="optimize the frozen geometry pool with bounded joint search")
    args = parser.parse_args()
    if args.joint and args.target_rows is not None:
        parser.error("--joint and --target-rows are mutually exclusive")
    if args.target_rows is not None and args.target_rows < 0:
        parser.error("--target-rows must be nonnegative")
    if args.output.resolve() in {args.source.resolve(), args.cover.resolve()}:
        parser.error("--output must differ from both inputs")

    try:
        source = read_pla(args.source)
        cover = read_pla(args.cover)
        if args.joint:
            rows, status = joint_recombine(source, cover)
            print(f"Joint cover {json.dumps(status, sort_keys=True)}")
        else:
            rows = recombine(source, cover, args.target_rows)
        write_pla(args.output, source.ninputs, source.noutputs, rows)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    before = sum(values.count("1") for _, values in cover.rows)
    after = sum(values.count("1") for _, values in rows)
    print(f"Merged rows {len(cover.rows)} -> {len(rows)}")
    print(f"Output connections {before} -> {after}")


if __name__ == "__main__":
    main()
