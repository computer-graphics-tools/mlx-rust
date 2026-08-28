//! Shape manipulation.

use mlx_rust_macros::{default_device, generate_macro};

use super::macros::unary_ops;
use crate::{
    array::{Array, VectorArray},
    error::Result,
    stream::Stream,
};

unary_ops! {
    /// View with at least one dimension.
    atleast_1d_device => mlx_atleast_1d,
    /// View with at least two dimensions.
    atleast_2d_device => mlx_atleast_2d,
    /// View with at least three dimensions.
    atleast_3d_device => mlx_atleast_3d,
    /// A copy of the array.
    copy_device => mlx_copy,
    /// Remove all length-1 axes.
    squeeze_device => mlx_squeeze,
    /// Reverse the order of the axes.
    transpose_device => mlx_transpose,
    /// Zeros with the same shape and dtype.
    zeros_like_device => mlx_zeros_like,
    /// Ones with the same shape and dtype.
    ones_like_device => mlx_ones_like,
}

/// A view of `a` with the given shape.
#[generate_macro]
#[default_device]
pub fn reshape_device(
    a: impl AsRef<Array>,
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_reshape(
            result,
            a.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().handle,
        )
    })
}

/// Broadcast `a` to the given shape.
#[generate_macro]
#[default_device]
pub fn broadcast_to_device(
    a: impl AsRef<Array>,
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_broadcast_to(
            result,
            a.as_ref().handle,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().handle,
        )
    })
}

/// Insert length-1 axes at `axes`.
#[generate_macro]
#[default_device]
pub fn expand_dims_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_expand_dims_axes(
            result,
            a.as_ref().handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Remove the length-1 axes at `axes`.
#[generate_macro]
#[default_device]
pub fn squeeze_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_squeeze_axes(
            result,
            a.as_ref().handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Permute the axes into the given order.
#[generate_macro]
#[default_device]
pub fn transpose_axes_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_transpose_axes(
            result,
            a.as_ref().handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Exchange two axes.
#[generate_macro]
#[default_device]
pub fn swapaxes_device(
    a: impl AsRef<Array>,
    axis1: i32,
    axis2: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_swapaxes(
            result,
            a.as_ref().handle,
            axis1,
            axis2,
            stream.as_ref().handle,
        )
    })
}

/// Move `source` to `destination`.
#[generate_macro]
#[default_device]
pub fn moveaxis_device(
    a: impl AsRef<Array>,
    source: i32,
    destination: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_moveaxis(
            result,
            a.as_ref().handle,
            source,
            destination,
            stream.as_ref().handle,
        )
    })
}

/// Collapse the axes from `start_axis` to `end_axis` into one.
#[generate_macro]
#[default_device]
pub fn flatten_device(
    a: impl AsRef<Array>,
    #[optional] start_axis: impl Into<Option<i32>>,
    #[optional] end_axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_flatten(
            result,
            a.as_ref().handle,
            start_axis.into().unwrap_or(0),
            end_axis.into().unwrap_or(-1),
            stream.as_ref().handle,
        )
    })
}

/// Join `arrays` along `axis`.
#[generate_macro]
#[default_device]
pub fn concatenate_device(
    arrays: &[&Array],
    #[optional] axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let mut vector = VectorArray::new();
    for array in arrays {
        vector.push(array)?;
    }
    let axis = axis.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_concatenate_axis(
            result,
            vector.handle(),
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Stack `arrays` along a new axis.
#[generate_macro]
#[default_device]
pub fn stack_device(
    arrays: &[&Array],
    #[optional] axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let mut vector = VectorArray::new();
    for array in arrays {
        vector.push(array)?;
    }
    let axis = axis.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_stack_axis(
            result,
            vector.handle(),
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Split into `num_splits` equal parts along `axis`.
#[generate_macro]
#[default_device]
pub fn split_device(
    a: impl AsRef<Array>,
    num_splits: i32,
    #[optional] axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let axis = axis.into().unwrap_or(0);
    let mut parts = VectorArray::new();
    crate::error::check(|| unsafe {
        mlx_rust_sys::mlx_split(
            parts.handle_mut(),
            a.as_ref().handle,
            num_splits,
            axis,
            stream.as_ref().handle,
        )
    })?;
    (0..parts.len()).map(|index| parts.get(index)).collect()
}
