//! Matrix products beyond `matmul`.

use mlx_rust_macros::{default_device, generate_macro};

use super::optional::optional_float;
use crate::{
    array::{Array, VectorArray, null_array},
    error::{Error, Result},
    stream::Stream,
};

/// `alpha * (a @ b) + beta * c`.
#[generate_macro]
#[default_device]
pub fn addmm_device(
    c: impl AsRef<Array>,
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] alpha: impl Into<Option<f32>>,
    #[optional] beta: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let c = c.as_ref();
    let a = a.as_ref();
    let b = b.as_ref();
    let alpha = alpha.into().unwrap_or(1.0);
    let beta = beta.into().unwrap_or(1.0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_addmm(
            result,
            c.handle,
            a.handle,
            b.handle,
            alpha,
            beta,
            stream.as_ref().handle,
        )
    })
}

/// Tensor contraction over `axes_a` and `axes_b`.
#[generate_macro]
#[default_device]
pub fn tensordot_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    axes_a: &[i32],
    axes_b: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_tensordot(
            result,
            a.handle,
            b.handle,
            axes_a.as_ptr(),
            axes_a.len(),
            axes_b.as_ptr(),
            axes_b.len(),
            stream.as_ref().handle,
        )
    })
}

/// Tensor contraction over the last `axis` dimensions.
#[generate_macro]
#[default_device]
pub fn tensordot_axis_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_tensordot_axis(
            result,
            a.handle,
            b.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Matmul with per-batch gathering.
#[generate_macro]
#[default_device]
pub fn gather_mm_device<'a>(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] lhs_indices: impl Into<Option<&'a Array>>,
    #[optional] rhs_indices: impl Into<Option<&'a Array>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let lhs_indices = lhs_indices.into();
    let rhs_indices = rhs_indices.into();
    let sorted_indices = sorted_indices.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_gather_mm(
            result,
            a.handle,
            b.handle,
            lhs_indices.map(|array| array.handle).unwrap_or_else(null_array),
            rhs_indices.map(|array| array.handle).unwrap_or_else(null_array),
            sorted_indices,
            stream.as_ref().handle,
        )
    })
}

/// Block-masked matmul.
#[generate_macro]
#[default_device]
pub fn block_masked_mm_device<'a>(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    block_size: i32,
    #[optional] mask_out: impl Into<Option<&'a Array>>,
    #[optional] mask_lhs: impl Into<Option<&'a Array>>,
    #[optional] mask_rhs: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let mask_out = mask_out.into();
    let mask_lhs = mask_lhs.into();
    let mask_rhs = mask_rhs.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_block_masked_mm(
            result,
            a.handle,
            b.handle,
            block_size,
            mask_out.map(|array| array.handle).unwrap_or_else(null_array),
            mask_lhs.map(|array| array.handle).unwrap_or_else(null_array),
            mask_rhs.map(|array| array.handle).unwrap_or_else(null_array),
            stream.as_ref().handle,
        )
    })
}

/// Segmented matmul.
#[generate_macro]
#[default_device]
pub fn segmented_mm_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    segments: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let segments = segments.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_segmented_mm(
            result,
            a.handle,
            b.handle,
            segments.handle,
            stream.as_ref().handle,
        )
    })
}

/// Hadamard transform.
#[generate_macro]
#[default_device]
pub fn hadamard_transform_device(
    a: impl AsRef<Array>,
    #[optional] scale: impl Into<Option<f32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let scale = optional_float(scale.into());
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_hadamard_transform(
            result,
            a.handle,
            scale,
            stream.as_ref().handle,
        )
    })
}

/// Einstein summation over `operands`.
#[generate_macro]
#[default_device]
pub fn einsum_device(
    subscripts: &str,
    operands: &[&Array],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let mut operands_vector = VectorArray::new();
    for array in operands {
        operands_vector.push(array)?;
    }
    let subscripts_cstring = std::ffi::CString::new(subscripts)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_einsum(
            result,
            subscripts_cstring.as_ptr(),
            operands_vector.handle(),
            stream.as_ref().handle,
        )
    })
}
