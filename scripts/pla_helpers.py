"""PLA I/O, care-table validation, and merge helpers for public IPC tools."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Pla:
    ninputs: int
    noutputs: int
    rows: list[tuple[str, str]]


class ExplicitPointValidator:
    """Validate cube covers with row bitsets over the explicit care table."""

    def __init__(self, source: Pla) -> None:
        self.source = source
        self.row_count = len(source.rows)
        self.all_rows = (1 << self.row_count) - 1
        byte_count = (self.row_count + 7) // 8

        input_bits = [
            [bytearray(byte_count), bytearray(byte_count)]
            for _ in range(source.ninputs)
        ]
        on_bits = [bytearray(byte_count) for _ in range(source.noutputs)]
        for row_index, (point, values) in enumerate(source.rows):
            byte_index = row_index >> 3
            bit = 1 << (row_index & 7)
            for variable, value in enumerate(point):
                input_bits[variable][value == "1"][byte_index] |= bit
            for output, value in enumerate(values):
                if value == "1":
                    on_bits[output][byte_index] |= bit

        self.input_bits = [
            (
                int.from_bytes(zero, "little"),
                int.from_bytes(one, "little"),
            )
            for zero, one in input_bits
        ]
        self.on_bits = [int.from_bytes(bits, "little") for bits in on_bits]
        self.coverage_cache: dict[str, int] = {}

    def cube_coverage(self, pattern: str) -> int:
        cached = self.coverage_cache.get(pattern)
        if cached is not None:
            return cached
        if len(pattern) != self.source.ninputs:
            raise ValueError(f"cube width {len(pattern)} != {self.source.ninputs}")
        covered = self.all_rows
        for variable, value in enumerate(pattern):
            if value == "0":
                covered &= self.input_bits[variable][0]
            elif value == "1":
                covered &= self.input_bits[variable][1]
            elif value != "-":
                raise ValueError(f"invalid cube character {value!r}")
            if covered == 0:
                break
        self.coverage_cache[pattern] = covered
        return covered

    def validate_output(self, output: int, patterns: Iterable[str]) -> tuple[bool, str]:
        covered = 0
        try:
            for pattern in patterns:
                covered |= self.cube_coverage(pattern)
        except ValueError as error:
            return False, str(error)
        required = self.on_bits[output]
        if required & ~covered:
            return False, "uncovered-on-row"
        if covered & (self.all_rows ^ required):
            return False, "covered-off-row"
        return True, "ok"

    def validate_all(self, outputs: list[list[str]]) -> tuple[bool, str]:
        if len(outputs) != self.source.noutputs:
            return False, "output-count-mismatch"
        for output, patterns in enumerate(outputs):
            valid, reason = self.validate_output(output, patterns)
            if not valid:
                return False, f"output-{output + 1:03d}:{reason}"
        return True, "ok"


def read_pla(path: Path) -> Pla:
    ninputs = noutputs = 0
    rows: list[tuple[str, str]] = []
    with path.open(encoding="ascii") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == ".i":
                ninputs = int(fields[1])
            elif fields[0] == ".o":
                noutputs = int(fields[1])
            elif not fields[0].startswith(".") and len(fields) >= 2:
                rows.append((fields[0], fields[1]))
    if ninputs <= 0 or noutputs <= 0:
        raise ValueError(f"invalid PLA header: {path}")
    for pattern, values in rows:
        if len(pattern) != ninputs or len(values) != noutputs:
            raise ValueError(f"invalid PLA row width: {path}")
    return Pla(ninputs, noutputs, rows)


def cover_patterns(path: Path, expected_inputs: int) -> list[str]:
    pla = read_pla(path)
    if pla.ninputs != expected_inputs or pla.noutputs != 1:
        raise ValueError(f"unexpected single-output cover shape: {path}")
    return [pattern for pattern, values in pla.rows if values[0] == "1"]


def literal_count(patterns: Iterable[str]) -> int:
    return sum(sum(value != "-" for value in pattern) for pattern in patterns)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def merge_outputs(outputs: list[list[str]], ninputs: int) -> tuple[list[tuple[str, str]], dict[str, int]]:
    geometries: dict[str, list[str]] = {}
    connections = 0
    for output, patterns in enumerate(outputs):
        for pattern in patterns:
            values = geometries.setdefault(pattern, ["0"] * len(outputs))
            if values[output] != "1":
                values[output] = "1"
                connections += 1
    rows = sorted((pattern, "".join(values)) for pattern, values in geometries.items())
    input_literals = sum(literal_count([pattern]) for pattern in geometries)
    metrics = {
        "merged_product_rows": len(rows),
        "output_connections": connections,
        "sharing_savings": connections - len(rows),
        "shared_product_rows": sum(values.count("1") > 1 for _, values in rows),
        "max_output_fanout": max((values.count("1") for _, values in rows), default=0),
        "input_literals": input_literals,
    }
    return rows, metrics


def write_pla(path: Path, ninputs: int, noutputs: int, rows: list[tuple[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        handle.write(f".i {ninputs}\n.o {noutputs}\n.type f\n.p {len(rows)}\n")
        for pattern, values in rows:
            handle.write(f"{pattern} {values}\n")
        handle.write(".e\n")
    os.replace(temporary, path)
