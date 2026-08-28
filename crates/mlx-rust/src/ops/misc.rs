//! Rounding, softmax, fp8 conversion and window functions.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{Dtype, array::Array, error::Result, stream::Stream};

/// Round to `decimals` places.
#[generate_macro]
#[default_device]
pub fn round_device(
    a: impl AsRef<Array>,
    #[optional] decimals: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let decimals = decimals.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_round(
            result,
            a.handle,
            decimals,
            stream.as_ref().handle,
        )
    })
}

/// Softmax over the whole array.
#[generate_macro]
#[default_device]
pub fn softmax_device(
    a: impl AsRef<Array>,
    #[optional] precise: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let precise = precise.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_softmax(
            result,
            a.handle,
            precise,
            stream.as_ref().handle,
        )
    })
}

/// Softmax over `axes`.
#[generate_macro]
#[default_device]
pub fn softmax_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] precise: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let precise = precise.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_softmax_axes(
            result,
            a.handle,
            axes.as_ptr(),
            axes.len(),
            precise,
            stream.as_ref().handle,
        )
    })
}

/// Softmax over one `axis`.
#[generate_macro]
#[default_device]
pub fn softmax_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] precise: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let precise = precise.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_softmax_axis(
            result,
            a.handle,
            axis,
            precise,
            stream.as_ref().handle,
        )
    })
}

/// Convert to 8-bit float.
#[generate_macro]
#[default_device]
pub fn to_fp8_device(
    x: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_to_fp8(result, x.handle, stream.as_ref().handle)
    })
}

/// Convert from 8-bit float to `dtype`.
#[generate_macro]
#[default_device]
pub fn from_fp8_device(
    x: impl AsRef<Array>,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_from_fp8(
            result,
            x.handle,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// `vals` broadcast to `a`'s shape.
#[generate_macro]
#[default_device]
pub fn full_like_device(
    a: impl AsRef<Array>,
    vals: impl AsRef<Array>,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let vals = vals.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_full_like(
            result,
            a.handle,
            vals.handle,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// An `n` by `m` lower-triangular matrix of ones.
#[generate_macro]
#[default_device]
pub fn tri_device(
    n: i32,
    m: i32,
    k: i32,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_tri(
            result,
            n,
            m,
            k,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// Bartlett window of length `m`.
#[generate_macro]
#[default_device]
pub fn bartlett_device(
    m: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_bartlett(result, m, stream.as_ref().handle)
    })
}

/// Blackman window of length `m`.
#[generate_macro]
#[default_device]
pub fn blackman_device(
    m: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_blackman(result, m, stream.as_ref().handle)
    })
}

/// Hamming window of length `m`.
#[generate_macro]
#[default_device]
pub fn hamming_device(
    m: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_hamming(result, m, stream.as_ref().handle)
    })
}

/// Hann window of length `m`.
#[generate_macro]
#[default_device]
pub fn hanning_device(
    m: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_hanning(result, m, stream.as_ref().handle)
    })
}
