#pragma once

// Bindgen input. Scoped to the subset of mlx-c needed for quantized-GEMM
// benchmarking -- 9 of mlx-c's 26 public headers. Transitive includes
// (optional.h, string.h, half.h, ...) come along; the allowlist in
// build/bindings.rs trims everything that is not `mlx_*` / `MLX_*`.

#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/fast.h"
#include "mlx/c/memory.h"
#include "mlx/c/metal.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/transforms.h"
#include "mlx/c/vector.h"
#include "mlx/c/version.h"
