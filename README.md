# DESCRIPTION

CCubes is a Boolean minimizer designed to efficiently minimize functions with multiple outputs. For the moment, it only accepts a PLA file as an input and produces a PLA file as output.

The minimization process is extremely efficient with incompletely specified functions with many don't cares. It compares the ON set minterms with the OFF set minterms, and for this reasons it needs a ".type fr" PLA file.

It employs a bottom-up search strategy, starting from the simplest combinations of 1 input, 2 inputs etc. and gradually increases the complexity. After each such level, it solves the minterm matrix and evaluates whether deeper prime implicants can still change the result.

The default stopping policy is adaptive. A one-term cover stops immediately because no nonempty function can use fewer terms. Otherwise, at the first unchanged boundary value, CCubes examines private witnesses of the current cover. If a compatible witness pair has no jointly covering retained PI, the output is structurally delayed and automatically continues under the certified policy. Otherwise CCubes accepts the plateau as a fast heuristic exit. A negative pair warning is not a proof of global optimality because a delayed group of three or more rows can escape the pair screen.

For applications that require a certificate from the outset, `-c` selects exact certified stopping based on compatible ON-row supercubes and an incompatibility lower bound. It stops only when the returned cover meets a global lower bound or the certified generation horizon has been completed and the boundary solver has proved its cover optimal. Gurobi supplies an exact boundary directly; the hybrid solver can also certify a boundary when its lower and upper bounds meet.

The small `examples/certified_F2.pla` instance demonstrates the escalation. Its cover value is two at levels one and two, but a delayed pair is detected; the default continues, reaches the one-term cover at level three, and certifies it.

The optional `-g` switch prints the machine-readable `CCUBES_BLOCKING` observation and the resulting `CCUBES_ADAPTIVE` action without changing the default policy. The `model_union_bound` field is an ex-ante all-ON-pair bound under an auxiliary independent, uniform, with-replacement OFF-row model; it is not conditioned on the observed cover and is not a finite-sample certificate. It is reported as `NA` when more than one million ON pairs would make this explanatory statistic disproportionately expensive.

Both stopping policies currently require binary, fully specified input rows and nonempty ON and OFF sets for every output. Unsupported inputs are rejected rather than silently processed under an unjustified stopping rule. The certified static analysis is lazy in the default mode: ordinary unwarned outputs do not pay for the full compatible-pair horizon and incompatibility scan.

Minimum covering of the PI chart is NP-hard. The `-s1` option delegates this boundary problem to the Gurobi optimizer. For compilation, Gurobi's path is configured in the Makefile and may need to be adjusted for the local installation and operating system. At runtime, Gurobi searches for a valid license; academic licences are available from Gurobi.

The default `-s0` option selects CCubes's own bundled hybrid covering solver. It combines dominance presolve, Lagrangian bounds and reduced-cost fixing, and a bounded branch-and-bound search of the remaining core. The effort levels `-e0` to `-e2` trade time for stronger bounds within fixed search budgets. The hybrid solver may prove optimality when its lower and upper bounds meet, but exactness is not guaranteed on every run; use `-s1` with Gurobi when a proven minimum boundary cover is required.

If no weights are applied, the combination of prime implicants that cover the ON set minterms is the quickest exact method, roughly equivalent to `espresso -Dso` type of output, although it produces a much more efficient circuit especially with an exact optimization.

Two weighting options are available, for instance the default `-w1` for weight based on complexity levels (prime implicants with lower number of literals will be given more weight). The option `-w2` adds additional weight if a prime implicant is shared between multiple outputs.

The option `-p` aims to spend additional time collecting a pool of possible solutions from the solver and decide which solution is best by comparing their shared prime implicants with those from the other outputs. Activating this parameter to more than 1 solutions will automatically set the weighting to `-w2`. Its rationale is that the solver may select different shared prime implicants, even though all have an equal (weighted) contribution to optimality.

Unlike other minimizers like Espresso (usually single threaded), CCubes is scalable and can handle larger problem instances more efficiently. Where possible, it will use a parallel search process using available CPU cores. Theoretically, its scalability can be extended to distributed computing environments, allowing it to tackle even larger instances by using multiple machines.

Parallel search supports both OpenMP and pthreads. The build automatically uses OpenMP when it is available; otherwise it uses the pthread backend, which is enabled by default. If neither backend is available or enabled, CCubes falls back to serial execution, so users do not need to install OpenMP specifically.

There is a minimal help system integrated into CCubes. `-d` canonicalizes PI ordering for deterministic experiments. Users can enable diagnostic logging with `-dbg<level>`; this provides insights into the internal workings of the minimization process and can be useful for troubleshooting. The older `-debug<level>` spelling remains accepted for compatibility.

The debug levels are preliminary, and more detailed logging functionality will be added in the future.

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
                          0 (default) fastest, bounded strong finish
                          1 stronger bounds, more time
                          2 best bound mode, adaptive bundle portfolio
  -d                  : deterministic PI ordering
  -g                  : print the adaptive blocking diagnostic at the first plateau
  -c                  : require certified exact stopping
                          default: adaptive warning, then certify only when warned
  -p<number>          : decide from a pool of up to <number> equally optimal solutions
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
