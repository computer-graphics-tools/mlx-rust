#pragma once

// Bindgen input: mlx-c's umbrella header, so the raw layer covers all of it.
// The allowlist in build/bindings.rs trims everything that is not `mlx_*`/`MLX_*`.
#include "mlx/c/mlx.h"
