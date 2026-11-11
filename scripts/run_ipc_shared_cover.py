#!/usr/bin/env python3
"""Run fresh IPC solves and the shared-row post-pass under a whole-PLA budget."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from pla_helpers import (
    ExplicitPointValidator,
    atomic_json,
    cover_patterns,
    merge_outputs,
    read_pla,
    sha256,
    write_pla,
)


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    deadline = started + args.seconds
    solve_deadline = deadline - args.finish_reserve
    args.output.mkdir()
    source = read_pla(args.source)
    if not source.rows or any(
        set(point + values) - {"0", "1"} for point, values in source.rows
    ):
        raise ValueError("expected a nonempty, explicit binary care table")
    validator = ExplicitPointValidator(source)
    outputs = [[] for _ in range(source.noutputs)]
    work = []
    for output, on in enumerate(validator.on_bits):
        if on == validator.all_rows:
            outputs[output] = ["-" * source.ninputs]
        elif on:
            work.append((output, 35 + math.sqrt(on.bit_count())))
    remaining_weight = sum(weight for _, weight in work)
    records = []
    for output, weight in work:
        slot_start = time.monotonic()
        slot = (solve_deadline - slot_start) * weight / remaining_weight
        if slot <= 0:
            raise TimeoutError("solver allowance exhausted")
        remaining_weight -= weight
        stem = args.output / f"output-{output + 1:03d}"
        care = stem.with_suffix(".fr.pla")
        cover = stem.with_suffix(".pla")
        with care.open("w", encoding="ascii") as handle:
            handle.write(f".i {source.ninputs}\n.o 1\n.type fr\n")
            handle.writelines(f"{point} {values[output]}\n" for point, values in source.rows)
            handle.write(".e\n")
        child_seconds = slot_start + slot - time.monotonic()
        if child_seconds <= 0:
            raise TimeoutError(f"no search time left for output {output + 1}")
        command = [
            str(args.binary), "--route", "ipc", "--threads", str(args.threads),
            "--literal-polish", "0", "--ipc-time-limit", f"{child_seconds * 0.8:.9f}",
            str(care), str(cover),
        ]
        timed_out = False
        with stem.with_suffix(".log").open("w") as log:
            try:
                subprocess.run(command, stdout=log, stderr=log, timeout=child_seconds, check=True)
            except subprocess.TimeoutExpired:
                # The native scalar entry point atomically publishes validated incumbents.
                timed_out = True
        if not cover.is_file():
            raise ValueError(f"output {output + 1} did not publish a cover")
        patterns = cover_patterns(cover, source.ninputs)
        valid, reason = validator.validate_output(output, patterns)
        if not valid:
            raise ValueError(f"output {output + 1}: {reason}")
        outputs[output] = patterns
        records.append(dict(output=output + 1, cubes=len(patterns), timed_out=timed_out,
                            seconds=time.monotonic() - slot_start, command=command))
        print(f"Output {output + 1}: {len(patterns)} cubes", flush=True)

    rows, before = merge_outputs(outputs, source.ninputs)
    raw = args.output / "ipc-independent-merged.pla"
    write_pla(raw, source.ninputs, source.noutputs, rows)
    solve_finished = time.monotonic()
    result = args.output / "ipc-shared-merged.pla"
    try:
        postpass = subprocess.run([
            sys.executable, str(Path(__file__).with_name("recombine_shared_cover.py")),
            "--joint", "--source", str(args.source), "--cover", str(raw),
            "--output", str(result),
        ], timeout=max(0, deadline - time.monotonic() - (2 if args.checker else 1)),
            text=True, stdout=subprocess.PIPE, check=True)
        print(postpass.stdout, end="")
        status_line = next(line for line in postpass.stdout.splitlines() if line.startswith("Joint cover "))
        joint = json.loads(status_line.removeprefix("Joint cover "))
        joint["timed_out"] = False
    except subprocess.TimeoutExpired:
        # The merged scalar incumbents have already been validated.
        write_pla(result, source.ninputs, source.noutputs, rows)
        joint = {"timed_out": True, "optimal_for_frozen_pool": False}
    postpass_finished = time.monotonic()
    if args.checker:
        subprocess.run([str(args.checker), str(args.source), str(result)],
                       timeout=max(0, deadline - time.monotonic()), check=True)
    elapsed = time.monotonic() - started
    final = read_pla(result)
    return dict(
        budget_seconds=args.seconds, finish_reserve_seconds=args.finish_reserve,
        elapsed_seconds=elapsed, within_budget=elapsed <= args.seconds,
        solve_and_merge_seconds=solve_finished - started,
        postpass_seconds=postpass_finished - solve_finished,
        joint_cover=joint,
        independent_check_seconds=elapsed + started - postpass_finished,
        checker=str(args.checker) if args.checker else None, threads=args.threads,
        before=before, merged_rows=len(final.rows),
        output_connections=sum(values.count("1") for _, values in final.rows),
        input_sha256=sha256(args.source), binary_sha256=sha256(args.binary),
        cover_sha256=sha256(result), outputs=records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--finish-reserve", type=float, default=25)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--checker", type=lambda path: Path(path).resolve(),
                        help="independent checker executable accepting source and result paths")
    args = parser.parse_args()
    if not (0 < args.finish_reserve < args.seconds < math.inf) or args.threads < 1:
        parser.error("require positive threads and 0 < finish reserve < total seconds")
    try:
        summary = run(args)
        atomic_json(args.output / "result.json", summary)
    except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{summary['merged_rows']} merged rows in {summary['elapsed_seconds']:.3f}s")
    if not summary["within_budget"]:
        parser.exit(1, "whole-PLA budget exceeded\n")


if __name__ == "__main__":
    main()
