//! Reductions.
//!
//! mlx-c splits each of these into three symbols -- whole-array, single-axis and
//! multi-axis. They are presented here as one function taking an optional `axes`,
//! matching `mlx.core`.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{array::Array, error::Result, stream::Stream};

/// Sum along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn sum_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_sum_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_sum(result, array.handle, keepdims, stream.handle)
        }),
    }
}

/// Mean along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn mean_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_mean_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_mean(
                result,
                array.handle,
                keepdims,
                stream.handle,
            )
        }),
    }
}

/// Product along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn prod_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_prod_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_prod(
                result,
                array.handle,
                keepdims,
                stream.handle,
            )
        }),
    }
}

/// Maximum along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn max_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_max_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_max(result, array.handle, keepdims, stream.handle)
        }),
    }
}

/// Minimum along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn min_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_min_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_min(result, array.handle, keepdims, stream.handle)
        }),
    }
}

/// Whether all elements are true, along `axes` or overall.
#[generate_macro]
#[default_device]
pub fn all_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_all_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_all(result, array.handle, keepdims, stream.handle)
        }),
    }
}

/// Whether any element is true, along `axes` or overall.
#[generate_macro]
#[default_device]
pub fn any_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_any_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_any(result, array.handle, keepdims, stream.handle)
        }),
    }
}

/// `log(sum(exp(a)))`, numerically stable, along `axes` or overall.
#[generate_macro]
#[default_device]
pub fn logsumexp_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axes.into() {
        Some(axes) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_logsumexp_axes(
                result,
                array.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_logsumexp(
                result,
                array.handle,
                keepdims,
                stream.handle,
            )
        }),
    }
}

/// Index of the maximum along `axis`, or of the flattened array.
#[generate_macro]
#[default_device]
pub fn argmax_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axis.into() {
        Some(axis) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_argmax_axis(
                result,
                array.handle,
                axis,
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_argmax(
                result,
                array.handle,
                keepdims,
                stream.handle,
            )
        }),
    }
}

/// Index of the minimum along `axis`, or of the flattened array.
#[generate_macro]
#[default_device]
pub fn argmin_device(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let keepdims = keepdims.into().unwrap_or(false);
    let stream = stream.as_ref();
    match axis.into() {
        Some(axis) => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_argmin_axis(
                result,
                array.handle,
                axis,
                keepdims,
                stream.handle,
            )
        }),
        None => Array::try_from_op(|result| unsafe {
            mlx_rust_sys::mlx_argmin(
                result,
                array.handle,
                keepdims,
                stream.handle,
            )
        }),
    }
}

/// Median along `axes`, or over the whole array when `axes` is `None`.
#[generate_macro]
#[default_device]
pub fn median_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let all_axes: Vec<i32> = (0..array.ndim() as i32).collect();
    let axes = axes.into().unwrap_or(&all_axes);
    let keepdims = keepdims.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_median(
            result,
            array.handle,
            axes.as_ptr(),
            axes.len(),
            keepdims,
            stream.as_ref().handle,
        )
    })
}

/// Variance along `axes`, or over the whole array when `axes` is `None`.
///
/// `ddof` is the delta degrees of freedom, defaulting to 0.
#[generate_macro]
#[default_device]
pub fn var_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let all_axes: Vec<i32> = (0..array.ndim() as i32).collect();
    let axes = axes.into().unwrap_or(&all_axes);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_var_axes(
            result,
            array.handle,
            axes.as_ptr(),
            axes.len(),
            keepdims.into().unwrap_or(false),
            ddof.into().unwrap_or(0),
            stream.as_ref().handle,
        )
    })
}

/// Standard deviation along `axes`, or over the whole array when `axes` is `None`.
///
/// `ddof` is the delta degrees of freedom, defaulting to 0.
#[generate_macro]
#[default_device]
pub fn std_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axes: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    #[optional] ddof: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    let all_axes: Vec<i32> = (0..array.ndim() as i32).collect();
    let axes = axes.into().unwrap_or(&all_axes);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_std_axes(
            result,
            array.handle,
            axes.as_ptr(),
            axes.len(),
            keepdims.into().unwrap_or(false),
            ddof.into().unwrap_or(0),
            stream.as_ref().handle,
        )
    })
}
