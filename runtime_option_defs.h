/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

/* Included repeatedly with PUBLIC_OPTION defined. */
PUBLIC_OPTION(PROGRESSIVE_COLUMN_POOL_IN, "pool-input", PATH, 0, 0, NULL, "Load a validated single-output IPC pool")
PUBLIC_OPTION(PROGRESSIVE_COLUMN_POOL_OUT, "pool-output", PATH, 0, 0, NULL, "Save the validated single-output IPC pool")
PUBLIC_OPTION(PROGRESSIVE_GLOBAL_BOUND_SECONDS, "global-bound-time-limit", NUMBER, 0, DBL_MAX, NULL, "IPC global-bound budget in seconds; 0 disables")
PUBLIC_OPTION(PROGRESSIVE_LITERAL_POLISH_DEADLINE_SECONDS, "literal-polish-time-limit", NUMBER, 0, DBL_MAX, NULL, "Separate final literal-polish allowance in seconds; 0 unlimited")
PUBLIC_OPTION(PROGRESSIVE_MASTER, "ipc-master", CHOICE, 0, 0, "native", "IPC restricted-master backend: native")
PUBLIC_OPTION(PROGRESSIVE_MEMORY_MB, "memory-mb", INTEGER, 1, INT_MAX, NULL, "IPC process memory envelope in MiB")
PUBLIC_OPTION(PROGRESSIVE_NATIVE_EFFORT, "ipc-effort", INTEGER, 0, 2, NULL, "Native IPC effort: 0, 1, or 2")
PUBLIC_OPTION(PROGRESSIVE_PROFILE, "profile-pricing", INTEGER, 0, 1, NULL, "IPC pricing telemetry: 0 or 1")
PUBLIC_OPTION(PROGRESSIVE_TIME_LIMIT_SECONDS, "ipc-time-limit", NUMBER, 0, DBL_MAX, NULL, "Per-output soft wall target in seconds; 0 disables")
PUBLIC_OPTION(PROGRESSIVE_TERMINAL_PATIENCE_SECONDS, "ipc-terminal-patience", NUMBER, 0, DBL_MAX, NULL, "End the terminal reserve after this many seconds without a smaller cover; 0 spends the whole reserve")
PUBLIC_OPTION(PROGRESSIVE_TERMINAL_RESERVE_SECONDS, "ipc-terminal-reserve", NUMBER, 0, DBL_MAX, NULL, "Final seconds of --ipc-time-limit spent on sustained local search over the final pool; 0 disables")
