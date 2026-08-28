//! Comparisons, selection and NaN handling.

use mlx_rust_macros::{default_device, generate_macro};

use super::optional::optional_float;
use crate::{
    array::{Array, null_array},
    error::Result,
    stream::Stream,
};

/// Whether all elements are close within `rtol` and `atol`.
#[generate_macro]
#[default_device]
pub fn allclose_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] rtol: impl Into<Option<f64>>,
    #[optional] atol: impl Into<Option<f64>>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let rtol = rtol.into().unwrap_or(1e-5);
    let atol = atol.into().unwrap_or(1e-8);
    let equal_nan = equal_nan.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_allclose(
            result,
            a.handle,
            b.handle,
            rtol,
            atol,
            equal_nan,
            stream.as_ref().handle,
        )
    })
}

/// Elementwise closeness within `rtol` and `atol`.
#[generate_macro]
#[default_device]
pub fn isclose_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] rtol: impl Into<Option<f64>>,
    #[optional] atol: impl Into<Option<f64>>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let rtol = rtol.into().unwrap_or(1e-5);
    let atol = atol.into().unwrap_or(1e-8);
    let equal_nan = equal_nan.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_isclose(
            result,
            a.handle,
            b.handle,
            rtol,
            atol,
            equal_nan,
            stream.as_ref().handle,
        )
    })
}

/// Whether two arrays have the same shape and elements.
#[generate_macro]
#[default_device]
pub fn array_equal_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] equal_nan: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let equal_nan = equal_nan.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_array_equal(
            result,
            a.handle,
            b.handle,
            equal_nan,
            stream.as_ref().handle,
        )
    })
}

/// Select from `x` where `condition` is true, else from `y`.
///
/// Named `select` because `where` is a Rust keyword; this is `mx.where`.
#[generate_macro]
#[default_device]
pub fn select_device(
    condition: impl AsRef<Array>,
    x: impl AsRef<Array>,
    y: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let condition = condition.as_ref();
    let x = x.as_ref();
    let y = y.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_where(
            result,
            condition.handle,
            x.handle,
            y.handle,
            stream.as_ref().handle,
        )
    })
}

/// Clip to `[a_min, a_max]`; either bound may be `None`.
#[generate_macro]
#[default_device]
pub fn clip_device<'a>(
    a: impl AsRef<Array>,
    #[optional] a_min: impl Into<Option<&'a Array>>,
    #[optional] a_max: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let a_min = a_min.into();
    let a_max = a_max.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_clip(
            result,
            a.handle,
            a_min.map(|array| array.handle).unwrap_or_else(null_array),
            a_max.map(|array| array.handle).unwrap_or_else(null_array),
            stream.as_ref().handle,
        )
    })
}

/// Replace NaN with `nan` and the infinities with `posinf`/`neginf`.
#[generate_macro]
#[default_device]
pub fn nan_to_num_device(
    a: impl AsRef<Array>,
    #[optional] nan: impl Into<Option<f32>>,
    #[optional] posinf: impl Into<Option<f32>>,
    #[optional] neginf: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let nan = nan.into().unwrap_or(0.0);
    let posinf = optional_float(posinf.into());
    let neginf = optional_float(neginf.into());
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_nan_to_num(
            result,
            a.handle,
            nan,
            posinf,
            neginf,
            stream.as_ref().handle,
        )
    })
}
