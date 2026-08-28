//! Building mlx-c's optional scalars.

/// `mlx_optional_float`, whose `value` MLX ignores when `has_value` is false.
pub(crate) fn optional_float(
    value: Option<f32>
) -> mlx_rust_sys::mlx_optional_float {
    mlx_rust_sys::mlx_optional_float {
        value: value.unwrap_or(0.0),
        has_value: value.is_some(),
    }
}
