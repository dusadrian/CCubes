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

The ordinary executable retains the existing automatic projection/MMCS policy, certified complete-prime enumeration, hybrid effort profiles, bounded plateau probe, and automatic pool budgets. Removing an experimental override does not remove algorithms already used by the normal policy. Internal search budgets remain implementation constants; they have not been expanded into new public tuning options.

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
| `-g` | `--blocking-diagnostic` |
| `-c` | `--certify` |
| `-p` | `--pool` |
| `-l3600=run.chk` | `--time-limit 3600 --checkpoint run.chk` |
| `--checkpoint-every=600` | `--checkpoint-every 600` (both forms remain valid) |
| `-r=run.chk` | `--resume run.chk` |
| `-i1=run.chk` | `--inspect run.chk --inspect-level 1` |
| `-dbg2=debug.log`, `-debug2=debug.log` | `--debug 2 --debug-file debug.log` |
| `-h` | `--help` |

`--threads 0` now consistently means automatic thread selection; previously explicitly passing `-t0` selected one thread while omission selected available cores. Other default algorithm choices are preserved. Former permissive `atoi()` parsing no longer accepts text such as `-sabc` as solver 0 or `-w1junk` as weight 1. Empty file paths and values are rejected. Combined legacy arguments are decoded without modifying `argv`.

The parser lives in `cli.c`/`cli.h`. One option registry supplies name matching, value requirements, and help descriptions. Parsing does not open files. `main.c` receives explicit settings, and the lock profiler receives an explicit Boolean. This keeps future configuration additions reviewable in one place.

The writer uses checkpoint version 9. When resuming version 8, CCubes replays the interrupted generation level from task zero. Resume restores the saved word width, weights, solver, and pool settings. Supply invocation-specific controls again as needed. Ordinary runs fall back to the bundled hybrid solver when Gurobi is unavailable.

## IPC configuration

Ordinary solving remains the default. Use `--route ipc` to select the native solver or `--route auto` for automatic selection. The legacy route value `e0` means ordinary, and `--progressive-cover` selects IPC. IPC-only resource, pool, and finishing options require a run routed to IPC.

Use `--ipc-time-limit` for a soft per-output target, `--memory-mb` for the native memory budget, and `--log-file` for either route. The [complete argument reference](README.md#complete-argument-reference) lists the supported controls. Older experiment settings remain documented with their original development revisions.

## Build and dependency settings retained

Build configuration remains in the Makefile: compiler selection (`CC`, `CXX`), build mode (`MODE`), CPU flags (`CPU_FLAGS`), sanitizers (`SAN`), backend selection (`DISABLE_OMP`, `USE_PTHREAD`), and Gurobi discovery/license paths (`GUROBI_HOME`, `GUROBI_LIC`, `GUROBI_LIB`, `GUROBI_LIBNAME`, `GRB_LICENSE_FILE`). These configure compilation or an external dependency and are not hidden minimization experiments. They should remain build/dependency settings rather than executable arguments.

The remaining interface migration step is to update external scripts that use deprecated short options before deciding when to remove those aliases. Ignored historical experiments may still contain old names; they need explicit migration if reused.
