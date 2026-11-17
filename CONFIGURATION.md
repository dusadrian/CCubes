# Configuration cleanup and migration

The public solver uses formal command-line arguments. The table below records the four former ordinary environment controls. Native IPC uses the same explicit configuration model.

## Runtime environment audit

| Former setting | Decision | Replacement |
| --- | --- | --- |
| `CCUBES_DETERMINISTIC` | Keep reproducible PI ordering; remove environment lookup. | `--deterministic` |
| `CCUBES_LOCK_STATS` | Keep useful contention profiling; pass enablement explicitly to the profiler. | `--lock-stats` |
| `CCUBES_TEST_PI_GENERATOR` | Retire forced-generator overrides. | Generator comparisons live in CCubesIPC. |
| `CCUBES_TEST_PROJECTION_RWLS` | Retire experimental RWLS activation. | The ordinary projection RWLS test build lives in CCubesIPC. |

CCubes reads its application settings from command-line arguments. OpenMP and Gurobi still read their own runtime settings, including the Gurobi license configuration.

The ordinary executable retains the existing automatic projection/MMCS policy, certified complete-prime enumeration, hybrid effort profiles, the one-cube shortcut, and automatic pool budgets. Removing an experimental override does not remove algorithms already used by the normal policy. Internal search budgets remain implementation constants; they have not been expanded into new public tuning options.

## Command-line migration

Long names are canonical in help, examples, and maintained integration tests. Values can follow a space or `=`. Names must match exactly; Boolean switches reject values. Short spellings remain compatibility aliases and produce one deprecation warning per invocation (`-h` remains a quiet help alias).

| Previous spelling | Preferred spelling |
| --- | --- |
| `-b64` | `--bits-per-word 64` |
| `-t8` | `--threads 8` |
| `-w0`, `-w1`, `-w2` | `--weights none`, `--weights complexity`, `--weights sharing` |
| `-s0`, `-s1` | `--solver hybrid`, `--solver gurobi` |
| `-e0`, `-e1`, `-e2` | `--effort 0`, `--effort 1`, `--effort 2` |
| `-d` | `--deterministic` |
| `-c` | `--certify` |
| `-p` | `--pool` |
| `--checkpoint-every=600` | `--checkpoint-every 600` (both forms remain valid) |
| `-r=run.chk` | `--resume run.chk` |
| `-i1=run.chk` | `--inspect run.chk --inspect-level 1` |
| `-dbg2=debug.log`, `-debug2=debug.log` | `--debug 2 --debug-file debug.log` |
| `-h` | `--help` |

`--threads 0` now consistently means automatic thread selection; previously explicitly passing `-t0` selected one thread while omission selected available cores. Other default algorithm choices are preserved. Former permissive `atoi()` parsing no longer accepts text such as `-sabc` as solver 0 or `-w1junk` as weight 1. Empty file paths and values are rejected. Combined legacy arguments are decoded without modifying `argv`.

The parser lives in `cli.c`/`cli.h`. One option registry supplies name matching, value requirements, and help descriptions. Parsing does not open files. `main.c` receives explicit settings, and the lock profiler receives an explicit Boolean. This keeps future configuration additions reviewable in one place.

The writer uses checkpoint version 11. The reader accepts versions 8 through 11. When resuming version 8, CCubes replays the interrupted generation level from task zero. Resume restores the saved word width, weights, solver, and pool settings. Supply invocation-specific controls again as needed. Ordinary runs fall back to the bundled hybrid solver when Gurobi is unavailable.

For the ordinary hybrid solver, effort 2 starts each smaller-cover search serially. For hard targets, one worker retains the original search order and node allowance while the others share a separate node allowance of the same size. Both allowances carry across successive targets within each profile. The team needs at least three workers and its size follows `--threads`, the remaining search budget, and worker scratch space. Equal-size pool enumeration runs separately. At debug level 1, `finish_nodes` and `finish_extra_nodes` report the work in the two allowances. `finish_workers` reports the largest team used and `finish_active_workers` counts workers that expanded a node or found a cover in a search. When multiple workers find valid covers, the first one accepted supplies the witness.

## Systematic certification

Use `--certify-systematic` to request a proof and `--certification-time-limit SECONDS` to set a positive shared deadline for each output after minimization. The shared clock includes validation, pricing, search, and proof checking. Omitting the deadline uses separate stage clocks with defaults of 10 seconds for weighted search and checking and 2 seconds for SAT and checking.

`--checkpoint-after` and `--checkpoint-every` act during ordinary projection. A save-and-exit pauses minimization before certification begins. Reapply the certification options when resuming a checkpoint.

| Advanced options | Behavior |
| --- | --- |
| `--systematic-weighted-seconds`, `--systematic-native-sat-seconds` | Set individual stage times. Zero skips that stage. Setting both to zero runs initial validation and pricing only. |
| `--systematic-weighted-work` | Caps weighted stage work. Zero disables the cap. |
| `--systematic-verifier-seconds`, `--systematic-verifier-work` | Enable a separate checker budget and pause the weighted search clock while checking. The default checker time is one second when enabled. An explicit work limit of zero removes the checker-specific cap. Any combined work cap still applies. |
| `--systematic-certification-work` | Caps combined weighted and SAT work. Zero disables the cap. With SAT enabled, the weighted stage gets at most half of the remaining work. |
| `--systematic-strengthening`, `--systematic-descent` | SAT profiles use `anchored-rows` by default. Other profiles are `none`, `groups`, `ordering`, and `combined`. Target descent uses `fresh` by default or `reuse`. |

The individual stage time and work limits require stage timing. The combined work cap and the experimental SAT controls can accompany either timing mode.

## IPC configuration

Ordinary solving remains the default. Use `--route ipc` to select the native solver or `--route auto` for automatic selection. The legacy route value `e0` means ordinary, and `--progressive-cover` selects IPC. IPC-only resource, pool, and finishing options require a run routed to IPC.

`--ipc-time-target` sets the soft search target per output. The terminal reserve comes out of the base target. Exact pricing extends the target, and post-search global-bound calculation and literal polishing have separate allowances. Use `--memory-mb` for the native memory budget and `--log-file` for either route. The [complete argument reference](README.md#complete-argument-reference) lists the supported controls. Older experiment settings remain documented with their original development revisions.

## Build and dependency settings retained

Build configuration remains in the Makefile: compiler selection (`CC`, `CXX`), build mode (`MODE`), CPU flags (`CPU_FLAGS`), sanitizers (`SAN`), backend selection (`DISABLE_OMP`, `USE_PTHREAD`), and Gurobi discovery/license paths (`GUROBI_HOME`, `GUROBI_LIC`, `GUROBI_LIB`, `GUROBI_LIBNAME`, `GRB_LICENSE_FILE`). These configure compilation or an external dependency and are not hidden minimization experiments. They should remain build/dependency settings rather than executable arguments.

The remaining interface migration step is to update external scripts that use deprecated short options before deciding when to remove those aliases. Ignored historical experiments may still contain old names; they need explicit migration if reused.
