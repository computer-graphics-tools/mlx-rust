//! Cumulative reductions.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{array::Array, error::Result, stream::Stream};

/// Cumulative sum along `axis`.
#[generate_macro]
#[default_device]
pub fn cumsum_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let reverse = reverse.into().unwrap_or(false);
    let inclusive = inclusive.into().unwrap_or(true);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_cumsum(
            result,
            a.handle,
            axis,
            reverse,
            inclusive,
            stream.as_ref().handle,
        )
    })
}

/// Cumulative product along `axis`.
#[generate_macro]
#[default_device]
pub fn cumprod_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let reverse = reverse.into().unwrap_or(false);
    let inclusive = inclusive.into().unwrap_or(true);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_cumprod(
            result,
            a.handle,
            axis,
            reverse,
            inclusive,
            stream.as_ref().handle,
        )
    })
}

/// Cumulative maximum along `axis`.
#[generate_macro]
#[default_device]
pub fn cummax_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let reverse = reverse.into().unwrap_or(false);
    let inclusive = inclusive.into().unwrap_or(true);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_cummax(
            result,
            a.handle,
            axis,
            reverse,
            inclusive,
            stream.as_ref().handle,
        )
    })
}

/// Cumulative minimum along `axis`.
#[generate_macro]
#[default_device]
pub fn cummin_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let reverse = reverse.into().unwrap_or(false);
    let inclusive = inclusive.into().unwrap_or(true);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_cummin(
            result,
            a.handle,
            axis,
            reverse,
            inclusive,
            stream.as_ref().handle,
        )
    })
}

/// Cumulative `log(sum(exp(a)))` along `axis`.
#[generate_macro]
#[default_device]
pub fn logcumsumexp_device(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] reverse: impl Into<Option<bool>>,
    #[optional] inclusive: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let reverse = reverse.into().unwrap_or(false);
    let inclusive = inclusive.into().unwrap_or(true);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_logcumsumexp(
            result,
            a.handle,
            axis,
            reverse,
            inclusive,
            stream.as_ref().handle,
        )
    })
}
