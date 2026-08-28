//! Array constructors that do not start from host data.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{Dtype, array::Array, error::Result, stream::Stream};

fn dtype_or_f32(dtype: Option<Dtype>) -> mlx_rust_sys::mlx_dtype {
    mlx_rust_sys::mlx_dtype_(dtype.unwrap_or(Dtype::Float32) as u32)
}

/// Zeros of the given shape.
#[generate_macro]
#[default_device]
pub fn zeros_device(
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_zeros(
            result,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// Ones of the given shape.
#[generate_macro]
#[default_device]
pub fn ones_device(
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_ones(
            result,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// Values from `start` up to but not including `stop`, spaced by `step`.
#[generate_macro]
#[default_device]
pub fn arange_device(
    start: f64,
    stop: f64,
    #[optional] step: impl Into<Option<f64>>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_arange(
            result,
            start,
            stop,
            step.into().unwrap_or(1.0),
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// `num` evenly spaced values from `start` to `stop`, inclusive.
#[generate_macro]
#[default_device]
pub fn linspace_device(
    start: f64,
    stop: f64,
    #[optional] num: impl Into<Option<i32>>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linspace(
            result,
            start,
            stop,
            num.into().unwrap_or(50),
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// An array of `shape` filled with `values`, broadcast as needed.
#[generate_macro]
#[default_device]
pub fn full_device(
    shape: &[i32],
    values: impl AsRef<Array>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_full(
            result,
            shape.as_ptr(),
            shape.len(),
            values.as_ref().handle,
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// An `n` by `m` matrix with ones on the `k`th diagonal.
#[generate_macro]
#[default_device]
pub fn eye_device(
    n: i32,
    #[optional] m: impl Into<Option<i32>>,
    #[optional] k: impl Into<Option<i32>>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_eye(
            result,
            n,
            m.into().unwrap_or(n),
            k.into().unwrap_or(0),
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}

/// The `n` by `n` identity matrix.
#[generate_macro]
#[default_device]
pub fn identity_device(
    n: i32,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_identity(
            result,
            n,
            dtype_or_f32(dtype.into()),
            stream.as_ref().handle,
        )
    })
}
