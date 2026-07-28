# DESCRIPTION

CCubes is a Boolean minimizer designed to efficiently minimize functions with multiple outputs. For the moment, it only accepts a PLA file as an input and produces a PLA file as output.

The minimization process is extremely efficient with incompletely specified functions with many don't cares. It compares the ON set minterms with the OFF set minterms, and for this reasons it needs a ".type fr" PLA file.

It employs a bottom-up search strategy, starting from the simplest combinations of 1 input, 2 inputs etc. and gradually increases the complexity. After each such level, it solves the minterm matrix and evaluates whether deeper prime implicants can still change the result.

The default `-s0 -e0` profile is deliberately the fastest hybrid heuristic. Before stopping at the first unchanged cover cardinality, it runs one small bounded plateau probe; if the probe does not improve the cover, `e0` stops and never triggers automatic certification. The optional `-g` diagnostic may describe the plateau, but it does not change this policy. An explicit `-c` remains an override for users who deliberately request certified exact stopping.

For fully specified binary point rows, `-e1`, `-e2`, and the Gurobi boundary profile use bounded adaptive stopping unless `-c` is requested. A one-term cover stops immediately because no nonempty function can use fewer terms. Before any hybrid effort profile accepts its first unchanged boundary value, CCubes performs one bounded plateau probe. It pairs private witnesses from different incumbent terms, maintains exact OFF-row blocker counts while expanding their agreement cubes, and prefers safe literal deletions that admit the most additional ON rows. Opposite deterministic tie orders provide candidate diversity. The probe adds only non-dominated prime candidates. The `-e0` and `-e1` budget examines at most 128 pairs and appends at most 32 candidates; `-e2` raises those caps to 1024 and 128. Candidates are retained only when re-solving the chart reduces cover cardinality; otherwise CCubes rolls them back. Probe exhaustion is not a certificate.

If the probe does not improve the cover, CCubes applies the existing adaptive diagnostic. Because this warning is cover-dependent, CCubes first checks the retained equal-cardinality solution pool and substitutes a warning-free tied cover when one exists. Only an unresolved warning can trigger deeper generation. That escalation is allowed only when the complete remaining certified horizon contains at most one million position-subset tasks; otherwise CCubes keeps the plateau cover, emits `action=warn-stop`, and finishes. If the bounded horizon is reached but the hybrid boundary solver has not closed its proof gap, it likewise warns and stops instead of searching past the budget. A negative pair warning is not a proof of global optimality because a delayed group of three or more rows can escape the pair screen.

For applications that require a certificate from the outset, `-c` selects exact certified stopping based on compatible ON-row supercubes and an incompatibility lower bound. It stops only when the returned cover meets a global lower bound or the certified generation horizon has been completed and the boundary solver has proved its cover optimal. Gurobi supplies an exact boundary directly; the hybrid solver can also certify a boundary when its lower and upper bounds meet.

The small `examples/certified_F2.pla` instance demonstrates the plateau probe under every hybrid effort profile. Its ordinary boundary cover has two terms at levels one and two. At the level-two plateau, the probe constructs the delayed prime and obtains the one-term cover without enumerating level three, including under the default `-e0` profile.

The optional `-g` switch prints the machine-readable `CCUBES_BLOCKING` observation without changing the selected stopping policy. Adaptive profiles also print the resulting `CCUBES_ADAPTIVE` action, and pool inspection is reported as `CCUBES_ADAPTIVE_POOL`. Under `-e0`, the observation remains diagnostic only. Unresolved adaptive warnings are always printed, even without `-g`, so that `action=certify` and `action=warn-stop` cannot be silent. The `model_union_bound` field is an ex-ante all-ON-pair bound under an auxiliary independent, uniform, with-replacement OFF-row model; it is not conditioned on the observed cover and is not a finite-sample certificate. It is reported as `NA` when more than one million ON pairs would make this explanatory statistic disproportionately expensive.

Input-dash PLA pattern rows are accepted through a separate, explicitly announced heuristic path, provided every output has nonempty ON and OFF sets. That path uses the first plateau and does not claim the point-row blocking diagnostic or a global certificate; `-c` is rejected. Certified static analysis is lazy in the adaptive point-row modes: ordinary unwarned outputs do not pay for the full compatible-pair horizon and incompatibility scan, while `-e0` never requests that analysis automatically.

Minimum covering of the PI chart is NP-hard. The `-s1` option delegates this boundary problem to the Gurobi optimizer. For compilation, Gurobi's path is configured in the Makefile and may need to be adjusted for the local installation and operating system. At runtime, Gurobi searches for a valid license; academic licences are available from Gurobi.

The default `-s0` option selects CCubes's own bundled hybrid covering solver. It combines dominance presolve, Lagrangian bounds and reduced-cost fixing, and a bounded branch-and-bound search of the remaining core. The effort levels `-e0` to `-e2` trade time for stronger bounds within fixed search budgets. The hybrid solver may prove optimality when its lower and upper bounds meet, but exactness is not guaranteed on every run; use `-s1` with Gurobi when a proven minimum boundary cover is required.

If no weights are applied, the combination of prime implicants that cover the ON set minterms is the quickest exact method, roughly equivalent to `espresso -Dso` type of output, although it produces a much more efficient circuit especially with an exact optimization.

Two weighting options are available, for instance the default `-w1` for weight based on complexity levels (prime implicants with lower number of literals will be given more weight). The option `-w2` adds additional weight if a prime implicant is shared between multiple outputs.

The Boolean option `-p` enables equal-cardinality cover pooling. The ordinary boundary solver always runs first, so pooling cannot replace its incumbent with a worse local cover. Alternative-cover discovery is deferred until the boundary remains on a plateau after the bounded plateau probe; if the probe lowers cardinality, CCubes keeps the improved ordinary incumbent and skips that pool solve. On a retained plateau, CCubes chooses a small candidate-discovery budget automatically from the observed PI-chart width: it grows logarithmically and is capped at 20 covers per output. Exact-boundary pooling collects alternative incumbents opportunistically instead of proving a complete ranking of the pool. The preferred incumbent and up to ten solver-ranked seed covers are preserved; remaining candidates are prioritized by marginal cross-output value. A cover is useful when it improves the best cube overlap available to candidate covers of another output, while near-duplicates have diminishing priority. Within the hard cap, CCubes conservatively retains any cover containing an exact cube match in another output's candidate landscape, because higher-order coordination can make such a cover useful even when its pairwise marginal score is zero. The selected discovery limit and generated/valuable/discarded counts are reported in diagnostic output, so experiments remain reproducible without asking users to guess a pool size.

After all outputs have reached their stopping boundary, CCubes performs one final coordination pass over every retained final-level pool, including outputs that stopped earlier. It selects one cover per output whose union contains as few distinct cube rows as possible. Among unions with the same number of rows, it then selects the one with the fewest exact input literals; each output's cheapest-literal tied cover is preserved during pool compaction so this secondary choice is not lost before coordination. The Cartesian product is searched exactly when it contains at most two million tuples; larger products use a deterministic three-start coordinate search. This is an optional lexicographic quality objective over the available candidates, not a proof of the globally smallest shared cover outside the retained pool. With the bundled hybrid solver, “equal-cardinality” means equal to its best feasible cover and is globally optimal only when the reported lower and upper bounds meet. Activating `-p` automatically selects `-w2`.

Unlike other minimizers like Espresso (usually single threaded), CCubes is scalable and can handle larger problem instances more efficiently. Where possible, it will use a parallel search process using available CPU cores. Theoretically, its scalability can be extended to distributed computing environments, allowing it to tackle even larger instances by using multiple machines.

Parallel search supports both OpenMP and pthreads. The build automatically uses OpenMP when it is available; otherwise it uses the pthread backend, which is enabled by default. If neither backend is available or enabled, CCubes falls back to serial execution, so users do not need to install OpenMP specifically.

PI coverage patterns from completed complexity levels are kept in a per-output hash index. Worker threads can therefore reject duplicate coverage in constant expected time without scanning the global PI chart under a lock. New PIs use only a short synchronized append, and coverage buckets are rebuilt once after the parallel level completes. This keeps the expensive generation phase parallel even when the PI chart contains tens of thousands of columns.

For fully specified binary point rows, CCubes automatically chooses between projection and an experimental bounded MMCS-style generator at each level. Small support spaces remain on projection. Once a level exceeds 8,192 projection tasks, CCubes gives MMCS a transactional trial of at most 8,192 search nodes per output, using at most four concurrent trial workers. A completed trial is retained; if any output reaches the budget, every trial append is rolled back to the common level boundary and projection completes the entire level. Thus partial MMCS work can never contaminate a boundary evaluation. Checkpoint/resume, time-limit checkpoints, and input-dash pattern rows remain on projection. The `--pi-generator=projection` and `--pi-generator=mmcs` options are expert overrides for reproducibility rather than choices expected from ordinary users.

The MMCS-style backend forms the OFF-row difference hypergraph for each ON row and enumerates exact-cardinality minimal transversals using an independently implemented uncovered-edge and critical-edge search. Every cube is assigned to the earliest ON row it covers, preventing repeated enumeration from later ON rows; complete ON coverage and cross-output sharing are rebuilt before the ordinary level-boundary evaluation. The included `examples/mmcs_forced_100x1.pla` fixture demonstrates the intended sparse-transversal regime: its sole PI has six literals, so the projection backend must pass through all 75,287,520 five-input supports before level six, whereas the bounded search reaches the forced level-six transversal directly. Difficult dense transversal families can still require exponential work, which is why automatic selection retains the projection fallback.

Current-level duplicate coverage is pruned after each generation level by default. During generation the coverage index therefore contains completed levels only and remains immutable, avoiding synchronized hash updates in the worker hot path. When pooling (`-p`) or sharing weights (`-w2`) are enabled, distinct shareable cube geometries are retained even when their local ON coverage is identical, because they may support different cross-output combinations. The optional `-d` switch additionally canonicalizes PI ordering for reproducible experiments; it does not enable or disable pruning.

Projection validation builds wildcard-aware OFF-row bit masks only for outputs whose OFF set contains input dashes. The same masks accelerate primality checks for shareable projected cubes. Fully specified outputs retain the decoded-index validation and scalar primality paths, so they do not pay the mask allocation or intersection cost.

There is a minimal help system integrated into CCubes. Users can enable diagnostic logging with `-dbg<level>`; this provides insights into the minimization process and can be useful for troubleshooting. The debug levels are preliminary, and more detailed logging functionality will be added in the future.

For very large problem instances, CCubes can save its state into a binary checkpoint file and exit, when a certain `-l` time limit is reached. The process can be resumed later from the checkpoint file, allowing users to continue the minimization process without re-specifying the input and output files. If the binary checkpoint file is not specified, it will default to `chk_<basename(source)>.bin`. Even when `-r`esuming from a checkpoint, a further time limit can be specified to save another intermediate checkpoint, and the binary checkpoint file will be overwritten unless specifying a different one. Checkpoint version 6 records adaptive/certified policy state; older checkpoint formats are intentionally not accepted.

The binary checkpoint file can be inspected using the `-i` option with various progress information in the metadata.

Compile the binary according to your system using `make`, with various options and customization indicated in the Makefile.

The destination .pla file is optional. If not specified, a `ccubes_<basename(source)>.pla` file will be created in the current directory.

# USAGE

```
ccubes [options] source.pla [dest.pla]
Options:
  -b<number>          : bits per word, either 8, 16, 32, 64 (default) or 128
  -t<number>          : number of CPU cores / threads to use with a parallel backend
  -w<number>          : weights applied to the prime implicants:
                          0 no weight
                          1 (default) weight based on complexity levels k
                          2 additional weight if shared between outputs
  -s<number>          : how to solve the covering problem:
                          0 (default) bundled hybrid solver
                            (presolve + Lagrangian bounds + bounded exact search)
                          1 Gurobi exact
  -e<number>          : hybrid solver effort level:
                          0 (default) small probe, then first-plateau stop
                          1 stronger bounds plus adaptive plateau handling
                          2 strongest bounds plus a larger plateau probe
  -d                  : deterministic PI ordering
  -g                  : print the adaptive blocking diagnostic at the first plateau
  -c                  : require certified exact stopping (point rows only)
                          explicitly overrides the -e0 heuristic plateau policy
                          input-dash rows: heuristic plateau stopping
  -p                  : enable automatic equal-cardinality cover pooling
  --pi-generator=<name> : auto (default), projection, or experimental mmcs
  -l<sec>[=<file>]    : time limit to save a checkpoint in the <file>
  -r=<file>           : resume from checkpoint file
  -i<level>=<file>    : inspect checkpoint (print progress and metadata)
                          0 (default) progress report
                          1 complete metadata about each output
  -dbg<level>[=<file>] : incremental debug information
                          0 (default) errors + warnings
                          1 errors + warnings + info
                          2 everything (trace)
  -h, --help          : show this help message
```
