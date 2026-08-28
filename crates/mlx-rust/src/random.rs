//! Pseudo-random arrays, mirroring `mlx.random`.
//!
//! Every generator takes an optional key. Passing `None` draws from MLX's global
//! state, which `seed` sets; passing a key from `key` or `split` makes the draw
//! reproducible and independent of that state.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    Dtype,
    array::{Array, null_array},
    error::{Result, check, install},
    stream::Stream,
};

fn dtype_or_f32(dtype: Option<Dtype>) -> mlx_rust_sys::mlx_dtype {
    mlx_rust_sys::mlx_dtype_(dtype.unwrap_or(Dtype::Float32) as u32)
}

fn key_handle(key: Option<&Array>) -> mlx_rust_sys::mlx_array {
    key.map(|key| key.handle).unwrap_or_else(null_array)
}

/// Seed MLX's global generator.
pub fn seed(seed: u64) -> Result<()> {
    check(|| unsafe { mlx_rust_sys::mlx_random_seed(seed) })
}

/// A key for reproducible draws.
pub fn key(seed: u64) -> Result<Array> {
    install();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_key(result, seed)
    })
}

/// Split `key` into two independent keys.
#[generate_macro]
#[default_device]
pub fn split_device(
    key: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let mut first = Array::empty();
    let mut second = Array::empty();
    check(|| unsafe {
        mlx_rust_sys::mlx_random_split(
            &mut first.handle,
            &mut second.handle,
            key.as_ref().handle,
            stream.as_ref().handle,
        )
    })?;
    Ok((first, second))
}

/// Split `key` into `num` independent keys, stacked along the first axis.
#[generate_macro]
#[default_device]
pub fn split_num_device(
    key: impl AsRef<Array>,
    num: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_split_num(
            result,
            key.as_ref().handle,
            num,
            stream.as_ref().handle,
        )
    })
}

/// Normal values with mean `loc` and standard deviation `scale`.
#[generate_macro]
#[default_device]
pub fn normal_device<'a>(
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] loc: impl Into<Option<f32>>,
    #[optional] scale: impl Into<Option<f32>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_normal(
            result,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            loc.into().unwrap_or(0.0),
            scale.into().unwrap_or(1.0),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}

/// Uniform values in `[low, high)`.
#[generate_macro]
#[default_device]
pub fn uniform_device<'a>(
    low: impl AsRef<Array>,
    high: impl AsRef<Array>,
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_uniform(
            result,
            low.as_ref().handle,
            high.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}

/// Integers in `[low, high)`.
#[generate_macro]
#[default_device]
pub fn randint_device<'a>(
    low: impl AsRef<Array>,
    high: impl AsRef<Array>,
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_randint(
            result,
            low.as_ref().handle,
            high.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(Some(dtype.into().unwrap_or(Dtype::Int32))),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}

/// Booleans that are true with probability `p`.
#[generate_macro]
#[default_device]
pub fn bernoulli_device<'a>(
    p: impl AsRef<Array>,
    shape: &[i32],
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_bernoulli(
            result,
            p.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}

/// Gumbel-distributed values.
#[generate_macro]
#[default_device]
pub fn gumbel_device<'a>(
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_gumbel(
            result,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}

/// Normal values truncated to `[lower, upper]`.
#[generate_macro]
#[default_device]
pub fn truncated_normal_device<'a>(
    lower: impl AsRef<Array>,
    upper: impl AsRef<Array>,
    shape: &[i32],
    #[optional] dtype: impl Into<Option<Dtype>>,
    #[optional] key: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let key = key.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_random_truncated_normal(
            result,
            lower.as_ref().handle,
            upper.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            dtype_or_f32(dtype.into()),
            key_handle(key),
            stream.as_ref().handle,
        )
    })
}
