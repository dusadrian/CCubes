/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_COVER_H
#define PROGRESSIVE_NATIVE_COVER_H

#include "progressive_cover.h"

ProgressiveCoverStatus progressive_native_cover_run(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result
);

#endif
