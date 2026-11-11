# Configuration cleanup and migration

This first cleanup covers the tracked executable sources, build rules, maintained tests, and README. There were four live application environment settings in that code. Older ignored experiments, benchmark archives, generated objects, and results are not part of the supported interface and have not been deleted.

## Runtime environment audit

| Former setting | Decision | Replacement |
| --- | --- | --- |
| `CCUBES_DETERMINISTIC` | Keep reproducible PI ordering; remove environment lookup. | `--deterministic` |
| `CCUBES_LOCK_STATS` | Keep useful contention profiling; pass enablement explicitly to the profiler. | `--lock-stats` |
| `CCUBES_TEST_PI_GENERATOR` | Remove forced-generator overrides from the production CLI; keep differential testing. | `--test-pi-generator` in the separate test executable only. |
| `CCUBES_TEST_PROJECTION_RWLS` | Remove experimental RWLS activation from production; retain the implementation and its regression evidence for research. | `--test-projection-rwls` in the separate test executable only. |

Neither executable reads these variables. The generic `env_flag_enabled()` helper was removed. `GRBgetenv(model)` is Gurobi's model-environment API, not a process-environment lookup, and remains unchanged. Standard runtime settings consumed by OpenMP and Gurobi, including license configuration, are outside the removed application switches.

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

Checkpoint versions and serialized layout are unchanged. Resuming restores the saved word width, weights, solver, and pool settings as before. Other invocation settings are not newly persisted by this cleanup. Automatic Gurobi fallback when unavailable also remains unchanged.

## Experimental code and tests

`make` builds `ccubes` from an explicit production source list. It does not link `projection_rwls.c`, and rejects both `--test-*` options. Test builds use separate objects under `.build/testing/`, so enabling tests cannot contaminate production objects.

```sh
make test
make .build/ccubes-testing
./.build/ccubes-testing --test-pi-generator projection --threads 1 \
  examples/certified_F2.pla /tmp/projection.pla
./.build/ccubes-testing --test-projection-rwls --deterministic --threads 4 \
  examples/boom_15x1x300.pla /tmp/rwls.pla
```

The test-only generator accepts `auto`, `projection`, `mmcs`, and `complete-mmcs`; the last requires `--certify`. These are research controls, not supported production arguments. RWLS stays disabled even in the test executable unless explicitly requested. Its existing integration test still checks the 27-to-25-term improvement fixture and the exact-boundary skip. Promoting RWLS into the default algorithm would require a separate performance and quality decision.

CLI regressions cover strict validation, long/legacy equivalence, word widths, Boolean semantics, paths containing spaces or `=`, option termination, parsing without log-file side effects, removed environment behavior, and rejection of test controls in production. Existing solver, certification, pooling, generator, and checkpoint tests remain in `make test`.

## Build and dependency settings retained

Build configuration remains in the Makefile: compiler selection (`CC`, `CXX`), build mode (`MODE`), CPU flags (`CPU_FLAGS`), sanitizers (`SAN`), backend selection (`DISABLE_OMP`, `USE_PTHREAD`), and Gurobi discovery/license paths (`GUROBI_HOME`, `GUROBI_LIC`, `GUROBI_LIB`, `GUROBI_LIBNAME`, `GRB_LICENSE_FILE`). These configure compilation or an external dependency and are not hidden minimization experiments. They should remain build/dependency settings rather than executable arguments.

The remaining interface migration step is to update external scripts that use deprecated short options before deciding when to remove those aliases. Ignored historical experiments may still contain old names; they need explicit migration if reused.
