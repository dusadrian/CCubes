# CCubes

## Ordinary solver

CCubes is a Boolean minimizer designed to efficiently minimize functions with multiple outputs. It reads and writes PLA files. The ordinary solver is the default. Choose native incremental prime cover (IPC) explicitly with `--route ipc`, or use `--route auto` to select a route from the input and the requested options.

The minimization process is extremely efficient with incompletely specified functions with many don't cares. It compares the ON set minterms with the OFF set minterms, and for this reasons it needs a ".type fr" PLA file.

It employs a bottom-up search strategy, starting from the simplest combinations of 1 input, 2 inputs etc. and gradually increases the complexity. After each such level, it solves the minterm matrix and evaluates whether deeper prime implicants can still change the result.

The default `--solver=hybrid --effort=0` profile is deliberately the fastest hybrid heuristic. Before stopping at the first unchanged cover cardinality, it runs one small bounded plateau probe; if the probe does not improve the cover, `--effort=0` stops and never triggers automatic certification. The optional `--blocking-diagnostic` diagnostic may describe the plateau, but it does not change this policy. An explicit `--certify` instead enumerates the complete prime chart and requires an exact cover of that chart.

For fully specified binary point rows, `--effort=1`, `--effort=2`, and the Gurobi boundary profile use the same heuristic plateau policy unless `--certify` is requested. A one-term cover stops immediately because no nonempty function can use fewer terms. Before any hybrid effort profile accepts its first unchanged boundary value, CCubes performs one bounded plateau probe. It pairs private witnesses from different incumbent terms, maintains exact OFF-row blocker counts while expanding their agreement cubes, and prefers safe literal deletions that admit the most additional ON rows. Opposite deterministic tie orders provide candidate diversity. The probe adds only non-dominated prime candidates. The `--effort=0` and `--effort=1` budget examines at most 128 pairs and appends at most 32 candidates; `--effort=2` raises those caps to 1024 and 128. Candidates are retained only when re-solving the chart reduces cover cardinality; otherwise CCubes rolls them back. Probe exhaustion is not a certificate.

If the probe does not improve the cover, CCubes applies the existing adaptive diagnostic. Because this warning is cover-dependent, CCubes first checks the retained equal-cardinality solution pool and substitutes a warning-free tied cover when one exists. An unresolved warning preserves the best heuristic cover; it no longer triggers the discarded support-horizon march. Run with `--certify` when a global certificate is required.

The small `examples/certified_F2.pla` instance demonstrates the plateau probe under every hybrid effort profile. Its ordinary boundary cover has two terms at levels one and two. At the level-two plateau, the probe constructs the delayed prime and obtains the one-term cover without enumerating level three, including under the default `--effort=0` profile.

Input-dash PLA pattern rows are accepted through a separate, explicitly announced heuristic path, with direct handling of constant outputs. That path uses the first plateau and does not claim the point-row blocking diagnostic or a global certificate; `--certify` is rejected.

Minimum covering of the PI chart is NP-hard. The `--solver=gurobi` option delegates this boundary problem to the Gurobi optimizer. For compilation, Gurobi's path is configured in the Makefile and may need to be adjusted for the local installation and operating system. At runtime, Gurobi searches for a valid license; academic licences are available from Gurobi.

The default `--solver=hybrid` option selects CCubes's own bundled hybrid covering solver. It combines dominance presolve, Lagrangian bounds and reduced-cost fixing, and a bounded branch-and-bound search of the remaining core. The effort levels `--effort=0` to `--effort=2` trade time for stronger bounds within fixed search budgets. The hybrid solver may prove optimality when its lower and upper bounds meet, but exactness is not guaranteed on every run; use `--solver=gurobi` with Gurobi when a proven minimum boundary cover is required.

If no weights are applied, the combination of prime implicants that cover the ON set minterms is the quickest exact method, roughly equivalent to `espresso -Dso` type of output, although it produces a much more efficient circuit especially with an exact optimization.

Two weighting options are available, for instance the default `--weights=complexity` for weight based on complexity levels (prime implicants with lower number of literals will be given more weight). The option `--weights=sharing` adds additional weight if a prime implicant is shared between multiple outputs.

The Boolean option `--pool` enables equal-cardinality cover pooling. The ordinary boundary solver always runs first, so pooling cannot replace its incumbent with a worse local cover. Alternative-cover discovery is deferred until the boundary remains on a plateau after the bounded plateau probe; if the probe lowers cardinality, CCubes keeps the improved ordinary incumbent and skips that pool solve. On a retained plateau, CCubes chooses a small candidate-discovery budget automatically from the observed PI-chart width: it grows logarithmically and is capped at 20 covers per output. Exact-boundary pooling collects alternative incumbents opportunistically instead of proving a complete ranking of the pool. The preferred incumbent and up to ten solver-ranked seed covers are preserved; remaining candidates are prioritized by marginal cross-output value. A cover is useful when it improves the best cube overlap available to candidate covers of another output, while near-duplicates have diminishing priority. Within the hard cap, CCubes conservatively retains any cover containing an exact cube match in another output's candidate landscape, because higher-order coordination can make such a cover useful even when its pairwise marginal score is zero. The selected discovery limit and generated/valuable/discarded counts are reported in diagnostic output, so experiments remain reproducible without asking users to guess a pool size.

After all outputs have reached their stopping boundary, CCubes performs one final coordination pass over every retained final-level pool, including outputs that stopped earlier. It selects one cover per output whose union contains as few distinct cube rows as possible. Among unions with the same number of rows, it then selects the one with the fewest exact input literals; each output's cheapest-literal tied cover is preserved during pool compaction so this secondary choice is not lost before coordination. The Cartesian product is searched exactly when it contains at most two million tuples; larger products use a deterministic three-start coordinate search. This is an optional lexicographic quality objective over the available candidates, not a proof of the globally smallest shared cover outside the retained pool. With the bundled hybrid solver, “equal-cardinality” means equal to its best feasible cover and is globally optimal only when the reported lower and upper bounds meet. Activating `--pool` automatically selects `--weights=sharing`.

Unlike other minimizers like Espresso (usually single threaded), CCubes is scalable and can handle larger problem instances more efficiently. Where possible, it will use a parallel search process using available CPU cores. Theoretically, its scalability can be extended to distributed computing environments, allowing it to tackle even larger instances by using multiple machines.

Parallel search supports both OpenMP and pthreads. The build automatically uses OpenMP when it is available; otherwise it uses the pthread backend, which is enabled by default. If neither backend is available or enabled, CCubes falls back to serial execution, so users do not need to install OpenMP specifically.

PI coverage patterns from completed complexity levels are kept in a per-output hash index. Worker threads can therefore reject duplicate coverage in constant expected time without scanning the global PI chart under a lock. New PIs use only a short synchronized append, and coverage buckets are rebuilt once after the parallel level completes. This keeps the expensive generation phase parallel even when the PI chart contains tens of thousands of columns.

For fully specified binary point rows in heuristic mode, CCubes automatically chooses between projection and an experimental bounded MMCS-style generator at each level. Small support spaces remain on projection. Once a level exceeds 8,192 projection tasks, CCubes screens the estimated branching cost and may give MMCS a transactional trial. Its node budget scales with the estimated projection work, with a floor of 8,192 and a ceiling of 100 million nodes per output, using at most four concurrent trial workers. A completed trial is retained; if any output reaches the budget, every trial append is rolled back to the common level boundary and projection completes the entire level. Thus partial MMCS work can never contaminate a boundary evaluation. Checkpointed runs and input-dash pattern rows remain on projection. Certified mode always uses complete MMCS.

The MMCS-style backend forms the OFF-row difference hypergraph for each ON row and enumerates exact-cardinality minimal transversals using an independently implemented uncovered-edge and critical-edge search. Every cube is assigned to the earliest ON row it covers, preventing repeated enumeration from later ON rows; complete ON coverage and cross-output sharing are rebuilt before the ordinary level-boundary evaluation. The included `examples/mmcs_forced_100x1.pla` fixture demonstrates the intended sparse-transversal regime: its sole PI has six literals, so the projection backend must pass through all 75,287,520 five-input supports before level six, whereas the bounded search reaches the forced level-six transversal directly. Difficult dense transversal families can still require exponential work, which is why automatic selection retains the projection fallback.

Current-level duplicate coverage is pruned after each generation level by default. During generation the coverage index therefore contains completed levels only and remains immutable, avoiding synchronized hash updates in the worker hot path. When pooling (`--pool`) or sharing weights (`--weights=sharing`) are enabled, distinct shareable cube geometries are retained even when their local ON coverage is identical, because they may support different cross-output combinations. The optional `--deterministic` switch additionally canonicalizes PI ordering for reproducible experiments; it does not enable or disable pruning.

Projection validation builds wildcard-aware OFF-row bit masks only for outputs whose OFF set contains input dashes. The same masks accelerate primality checks for shareable projected cubes. Fully specified outputs retain the decoded-index validation and scalar primality paths, so they do not pay the mask allocation or intersection cost.

There is a minimal help system integrated into CCubes. Users can enable diagnostic logging with `--debug LEVEL`; this provides insights into the minimization process and can be useful for troubleshooting. The debug levels are preliminary, and more detailed logging functionality will be added in the future.

For very large heuristic runs, CCubes can save its state into a binary checkpoint file and exit when a specified `--time-limit` time limit is reached. The process can be resumed later without re-specifying the input and output files. If the binary checkpoint file is not specified, it defaults to the source basename prefixed with `chk_`. Even when resuming with `--resume FILE`, another time limit can save a later intermediate checkpoint, overwriting the binary checkpoint unless a different path is given.

`--checkpoint-every=<sec>` adds periodic checkpoints without ending the process. At each interval, the projection workers stop at a safe shared frontier, CCubes atomically replaces the checkpoint file, and generation continues at the same complexity level. When it is combined with `--time-limit`, periodic checkpoints continue until the overall per-invocation time limit is reached; that final checkpoint still ends the process. For example, a seven-hour resumed run with hourly checkpoints is:

```sh
./ccubes --threads 16 --time-limit 25200 --checkpoint output096.chk \
  --checkpoint-every 3600 --resume output096.chk
```

Checkpoint polling occurs during interruptible projection generation. A long covering solve can delay a save until generation next reaches a safe point. Checkpoints apply to ordinary runs without complete-prime certification. Checkpoint version 9 stores the earliest unfinished task across workers. Resuming a version 8 file replays the interrupted level from task zero while retaining its saved chart. Each save writes to an exclusive temporary file beside the target and replaces the previous checkpoint after the write succeeds.

The binary checkpoint file can be inspected using the `--inspect` option with various progress information in the metadata.

Compile the binary according to your system using `make`, with various options and customization indicated in the Makefile.

The destination .pla file is optional. If not specified, the result uses the input basename prefixed by `ccubes_` for ordinary runs or `progressive_` for IPC.

## Usage

Use descriptive long options. Options taking values accept either `--name VALUE` or `--name=VALUE`. Boolean switches take no value. Use `--` before filenames beginning with a dash.

```sh
./ccubes --threads 8 --solver hybrid --effort 1 --pool source.pla result.pla
./ccubes --certify --deterministic source.pla result.pla
./ccubes --route ipc --threads 8 --ipc-time-limit 180 source.pla result.pla
./ccubes --inspect run.chk --inspect-level 1
```

## Complete argument reference

The following table lists all formal arguments printed by `./ccubes --help`.
The CCubesIPC development checks compare this table with the built executable's help. Update the reference when an argument, allowed value, or default changes.

```text
ccubes [options] source.pla [dest.pla]
ccubes --resume FILE [options]
ccubes --inspect FILE [--inspect-level 0|1]
```

Arguments with values accept either `--name VALUE` or `--name=VALUE`. A dash in the Value column means a Boolean switch that takes no value. `--` ends option parsing. Deprecated short aliases and their replacements are listed in [CONFIGURATION.md](CONFIGURATION.md#short-option-migration).

| Argument | Value | Meaning |
| --- | --- | --- |
| `--bits-per-word` | `8`, `16`, `32`, `64`, `128` | Ordinary PI storage word width; default `64`. |
| `--threads` | Integer ≥ 0 | CPU worker threads; `0` (default) selects available cores. |
| `--weights` | `none`, `complexity`, `sharing` | Ordinary cover weighting; default `complexity`. |
| `--solver` | `hybrid`, `gurobi` | Ordinary cover solver; default `hybrid`. |
| `--effort` | `0`, `1`, `2` | Ordinary hybrid solver effort; default `0`. |
| `--deterministic` | — | Canonicalize ordinary PI ordering; not a guarantee of identical time-limited or IPC results. |
| `--certify` | — | Complete-prime enumeration and exact cover for supported binary point inputs. |
| `--blocking-diagnostic` | — | Report the ordinary adaptive blocking diagnostic at the first plateau. |
| `--pool` | — | Enable bounded equal-cardinality cover pooling; implies sharing weights. |
| `--lock-stats` | — | Report PI-insertion lock contention. |
| `--time-limit` | Nonnegative seconds | Ordinary generation budget before checkpoint and exit; `0` disables. |
| `--checkpoint-every` | Nonnegative seconds | Interval between ordinary generation checkpoints. Generation continues after each save. `0` disables periodic saves. |
| `--checkpoint` | File path | Checkpoint save path. Requires a positive `--time-limit` or `--checkpoint-every`. If omitted, the path is derived from the source or resumed checkpoint. |
| `--resume` | File path | Resume an ordinary generation checkpoint. |
| `--inspect` | File path | Inspect a checkpoint and exit. |
| `--inspect-level` | `0`, `1` | Inspection detail: progress (`0`, default) or full metadata (`1`); requires `--inspect`. |
| `--debug` | `0`, `1`, `2` | Logging detail: warnings, information, or trace; logging is off unless requested. |
| `--debug-file` | File path | Write debug logging to this file; otherwise use standard error. |
| `--log-file` | File path | Append run diagnostics and solver logs for any backend and output count; defaults to debug level `1` unless `--debug` is supplied. |
| `--help` | — | Print usage and the complete argument list, then exit. |
| `--progressive-cover` | — | Select IPC; equivalent to `--route ipc`. |
| `--route` | `auto`, `ordinary`, `ipc` | Select the minimization pipeline; default `ordinary`. The legacy value `e0` means `ordinary`. |
| `--literal-polish` | `0`, `1`, `2`, `3` | IPC finishing: off, one pass (default), yield-gated passes, or four passes. |
| `--pool-input` | File path | Load a validated single-output IPC pool and continue normal minimization. |
| `--pool-output` | File path | Save the validated single-output IPC pool. |
| `--global-bound-time-limit` | Nonnegative seconds | Optional IPC global-bound budget; `0` disables. |
| `--literal-polish-time-limit` | Nonnegative seconds | Separate allowance for final IPC literal polishing; `0` means unlimited. |
| `--ipc-master` | `native` | Select the native IPC restricted master (default). |
| `--memory-mb` | Positive integer | IPC process memory envelope in MiB. |
| `--ipc-effort` | `0`, `1`, `2` | Native IPC master effort; default `0`. |
| `--profile-pricing` | `0`, `1` | IPC pricing telemetry; default off (`0`). |
| `--ipc-time-limit` | Nonnegative seconds | Soft per-output IPC wall target; `0` disables. Finishing work can exceed the target. |
| `--ipc-terminal-reserve` | Nonnegative seconds | Set aside part of the IPC target for sustained local search over the final pool. Default `0` disables it. A positive reserve requires a larger `--ipc-time-limit`. |
| `--ipc-terminal-patience` | Nonnegative seconds | End the reserve early after a period without a smaller cover. The default is `8` seconds, capped at the duration of the main IPC search. Use `0` to disable early stopping for lack of progress. |

The default solver remains the bundled hybrid solver, with effort 0 and complexity weights. Numeric solver values `0`/`1` and weights `0`/`1`/`2` are also accepted. `--threads 0` selects available cores, just like omitting the option. Invalid, empty, out-of-range, or partially numeric values are rejected before opening a debug log or starting minimization.

`--checkpoint FILE` selects the save path and requires a positive `--time-limit` or `--checkpoint-every`. `--inspect-level` requires `--inspect`. `--debug-file FILE` enables logging at level 0 unless `--debug` supplies another level. Repeated settings use the last value; `--pool` always implies sharing weights.

Resume retains the existing checkpoint behavior: saved word width, weights, solver, and pooling settings are restored from the file. Thread count, effort, diagnostics, deterministic ordering, time limit, and checkpoint interval are settings for the current invocation and should be supplied again when needed.

Short options still work as deprecated aliases during migration. The four former `CCUBES_*` environment switches are retired. See [CONFIGURATION.md](CONFIGURATION.md) for the migration table.


## Native IPC

Use `--route ipc` for the native incremental prime-cover solver. It handles constant outputs directly and solves and merges supported outputs independently. With `--route auto`, eligible wide or dense inputs can select IPC. Ordinary-only requests such as checkpoints, certification, pooling, and an explicit Gurobi solver keep the ordinary route when automatic routing is selected. The `CCUBES_ROUTE` line records the choice and reason.

IPC defaults to the native master at effort zero with greedy pricing, native repair, bounded recovery, and one final literal-polish pass. It runs without Gurobi. The ordinary route retains its optional Gurobi solver and falls back to the hybrid solver if Gurobi support was not built in. The IPC wall target is soft because current work and finalization may extend beyond it. On a single-output run, the solver can publish validated incumbents atomically as it progresses.

For example, `--route ipc --ipc-time-limit 60 --ipc-terminal-reserve 15` gives the main IPC search a 45-second target and reserves up to 15 seconds for local search over the retained columns. The reserve can finish early after the patience interval or when the retained pool already has a proven minimum cover. The patience check runs between search rounds.

Use `--pool-output` to retain a single-output IPC pool and `--pool-input` to validate and continue minimization from a compatible pool. These files have a different format from ordinary checkpoints. Restricted-master bounds and proofs apply to the retained pool. A global bound requires separate verification against the full pricing domain.

### Shared rows under a whole-PLA budget

The Python runner accepts an explicit binary care table and a new output directory. It runs native IPC on each nonconstant output and searches for a smaller merged cover using geometries already in the validated merged seed.

```sh
python3 scripts/run_ipc_shared_cover.py --binary ./ccubes \
  --source source.pla --output joint-result --threads 8 --seconds 180
```

The runner reserves finishing time inside the total budget. It retains the validated seed if the post-pass times out or would increase the merged row count. The optional `--checker` accepts an independent executable that validates the source and serialized result. The run records merged rows, output connections, and any optimum proved within the selected pool.

### Run logs

Use `--log-file FILE` to append diagnostics from either route. It defaults to information level. Set `--debug` to change the level or `--debug-file` to send debug output to a separate file. Keep logs separate from inputs, outputs, checkpoint and pool files. Help and checkpoint inspection still use standard output.

## Verification

The ordinary and IPC regression suites are maintained in the CCubesIPC development repository.
