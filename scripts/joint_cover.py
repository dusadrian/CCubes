"""Dependency-free optimization of merged rows in a frozen geometry pool."""

from pla_helpers import ExplicitPointValidator, Pla, merge_outputs


try:
    count_bits = int.bit_count
except AttributeError:
    # The external benchmark server uses Python 3.8.
    def count_bits(value):
        return bin(value).count("1")


def prepare_joint_pool(source: Pla, pool: Pla):
    if (source.ninputs, source.noutputs) != (pool.ninputs, pool.noutputs):
        raise ValueError("care table and geometry pool dimensions differ")
    if not source.rows or any(set(point + values) - {"0", "1"} for point, values in source.rows):
        raise ValueError("expected a nonempty, explicit binary care table")
    if any(set(mask) - {"0", "1"} for _, mask in pool.rows):
        raise ValueError("expected binary output masks in the geometry pool")
    validator = ExplicitPointValidator(source)
    patterns = [pattern for pattern, _ in pool.rows]
    coverage = [validator.cube_coverage(pattern) for pattern in patterns]
    if len(set(patterns)) != len(patterns):
        raise ValueError("expected one row per geometry")
    eligible = []
    forced = set()
    for output, on in enumerate(validator.on_bits, 1):
        legal = [i for i, mask in enumerate(coverage) if mask and not (mask & ~on)]
        eligible.append(legal)
        covered = repeated = 0
        for i in legal:
            repeated |= covered & coverage[i]
            covered |= coverage[i]
        if covered != on:
            raise ValueError(f"geometry pool cannot cover output {output}")
        # Every geometry that alone covers an ON point is mandatory.
        singleton_points = covered & ~repeated
        forced.update(i for i in legal if coverage[i] & singleton_points)

    edges = set()
    residual_points = 0
    for on, legal in zip(validator.on_bits, eligible):
        residual = on
        for i in legal:
            if i in forced:
                residual &= ~coverage[i]
        residual_points += count_bits(residual)
        by_point = {}
        for i in legal:
            hits = coverage[i] & residual
            while hits:
                low = hits & -hits
                by_point.setdefault(low.bit_length() - 1, []).append(i)
                hits ^= low
        if len(by_point) != count_bits(residual):
            raise ValueError("geometry pool cannot cover a residual point")
        edges.update(tuple(indices) for indices in by_point.values())
    return validator.on_bits, eligible, coverage, forced, edges, residual_points


def reconstruct_joint_cover(pool: Pla, required, eligible, coverage, chosen):
    patterns = [pattern for pattern, _ in pool.rows]
    # Output attachments do not affect the merged-row objective.
    outputs = []
    for o, (on, legal) in enumerate(zip(required, eligible)):
        usable = [i for i in legal if i in chosen]
        attached = [i for i in usable if pool.rows[i][1][o] == "1"]
        residual = on
        for i in attached:
            residual &= ~coverage[i]
        candidates = [i for i in usable if coverage[i] & residual]
        while residual:
            if not candidates:
                raise ValueError(f"solver selection cannot cover output {o + 1}")
            best = max(candidates, key=lambda i: count_bits(coverage[i] & residual))
            if not coverage[best] & residual:
                raise ValueError(f"solver selection cannot cover output {o + 1}")
            attached.append(best)
            residual &= ~coverage[best]
        outputs.append([patterns[i] for i in attached])
    return merge_outputs(outputs, pool.ninputs)


def solve_residual_cover(edges, node_limit=10_000):
    """Return a feasible selection and whether the search proved it optimal."""
    ordered = sorted(edges, key=lambda edge: (len(edge), edge))
    incidence = {}
    for index, edge in enumerate(ordered):
        for column in edge:
            incidence[column] = incidence.get(column, 0) | (1 << index)

    remaining = (1 << len(ordered)) - 1
    uncovered = remaining
    greedy = []
    while uncovered:
        index = (uncovered & -uncovered).bit_length() - 1
        column = max(ordered[index], key=lambda i: count_bits(incidence[i] & uncovered))
        greedy.append(column)
        uncovered &= ~incidence[column]
    best = tuple(greedy)

    # Branch on the first remaining requirement. Disjoint requirements each
    # need their own geometry, which also proves the frozen Fišer residual.
    stack = [(remaining, ())]
    for _ in range(node_limit):
        if not stack:
            return set(best), True
        uncovered, selected = stack.pop()
        if not uncovered:
            if len(selected) < len(best):
                best = selected
            continue

        needed = 0
        available = uncovered
        while available and len(selected) + needed < len(best):
            index = (available & -available).bit_length() - 1
            for column in ordered[index]:
                available &= ~incidence[column]
            needed += 1
        if len(selected) + needed >= len(best):
            continue

        index = (uncovered & -uncovered).bit_length() - 1
        choices = sorted(ordered[index], key=lambda i: count_bits(incidence[i] & uncovered))
        for column in choices:
            stack.append((uncovered & ~incidence[column], (*selected, column)))
    return set(best), not stack


def joint_recombine(source: Pla, pool: Pla, node_limit=10_000):
    required, eligible, coverage, forced, edges, residual_points = prepare_joint_pool(source, pool)
    original = [(pattern, mask) for pattern, mask in pool.rows if "1" in mask]
    for output, on in enumerate(required):
        covered = 0
        for (_, mask), points in zip(pool.rows, coverage):
            if mask[output] == "1":
                covered |= points
        if covered != on:
            raise ValueError(f"original cover disagrees with care table on output {output + 1}")
    selected, optimal = solve_residual_cover(edges, node_limit)
    rows, _ = reconstruct_joint_cover(pool, required, eligible, coverage, forced | selected)
    if optimal and len(rows) != len(forced) + len(selected):
        raise ValueError("serialized row count differs from the proved optimum")
    retained_original = len(rows) > len(original)
    if retained_original:
        rows = original
        optimal = False
    return rows, {
        "optimal_for_frozen_pool": optimal,
        "retained_original_cover": retained_original,
        "forced_geometries": len(forced),
        "residual_points": residual_points,
        "residual_constraints": len(edges),
    }
