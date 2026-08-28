//! Reshaping, padding, tiling and diagonals.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    Dtype,
    array::{Array, VectorArray},
    error::{Error, Result, check},
    stream::Stream,
};

/// Insert one length-1 axis at `axis`.
#[generate_macro]
#[default_device]
pub fn expand_dims_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_expand_dims(
            result,
            a.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Remove the length-1 axis at `axis`.
#[generate_macro]
#[default_device]
pub fn squeeze_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_squeeze_axis(
            result,
            a.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Expand `axis` into `shape`.
#[generate_macro]
#[default_device]
pub fn unflatten_device(
    a: impl AsRef<Array>,
    axis: i32,
    shape: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_unflatten(
            result,
            a.handle,
            axis,
            shape.as_ptr(),
            shape.len(),
            stream.as_ref().handle,
        )
    })
}

/// Roll elements by `shift`, wrapping around.
#[generate_macro]
#[default_device]
pub fn roll_device(
    a: impl AsRef<Array>,
    shift: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_roll(
            result,
            a.handle,
            shift.as_ptr(),
            shift.len(),
            stream.as_ref().handle,
        )
    })
}

/// Roll by `shift` along `axes`.
#[generate_macro]
#[default_device]
pub fn roll_axes_device(
    a: impl AsRef<Array>,
    shift: &[i32],
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_roll_axes(
            result,
            a.handle,
            shift.as_ptr(),
            shift.len(),
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Roll by `shift` along one `axis`.
#[generate_macro]
#[default_device]
pub fn roll_axis_device(
    a: impl AsRef<Array>,
    shift: &[i32],
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_roll_axis(
            result,
            a.handle,
            shift.as_ptr(),
            shift.len(),
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Repeat each element `repeats` times.
#[generate_macro]
#[default_device]
pub fn repeat_device(
    arr: impl AsRef<Array>,
    repeats: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let arr = arr.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_repeat(
            result,
            arr.handle,
            repeats,
            stream.as_ref().handle,
        )
    })
}

/// Repeat `repeats` times along `axis`.
#[generate_macro]
#[default_device]
pub fn repeat_axis_device(
    arr: impl AsRef<Array>,
    repeats: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let arr = arr.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_repeat_axis(
            result,
            arr.handle,
            repeats,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Tile by `reps` along each axis.
#[generate_macro]
#[default_device]
pub fn tile_device(
    arr: impl AsRef<Array>,
    reps: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let arr = arr.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_tile(
            result,
            arr.handle,
            reps.as_ptr(),
            reps.len(),
            stream.as_ref().handle,
        )
    })
}

/// Extract or construct a diagonal.
#[generate_macro]
#[default_device]
pub fn diag_device(
    a: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_diag(result, a.handle, k, stream.as_ref().handle)
    })
}

/// Diagonal over `axis1` and `axis2`.
#[generate_macro]
#[default_device]
pub fn diagonal_device(
    a: impl AsRef<Array>,
    #[optional] offset: impl Into<Option<i32>>,
    #[optional] axis1: impl Into<Option<i32>>,
    #[optional] axis2: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let offset = offset.into().unwrap_or(0);
    let axis1 = axis1.into().unwrap_or(0);
    let axis2 = axis2.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_diagonal(
            result,
            a.handle,
            offset,
            axis1,
            axis2,
            stream.as_ref().handle,
        )
    })
}

/// Sum along a diagonal.
#[generate_macro]
#[default_device]
pub fn trace_device(
    a: impl AsRef<Array>,
    #[optional] offset: impl Into<Option<i32>>,
    #[optional] axis1: impl Into<Option<i32>>,
    #[optional] axis2: impl Into<Option<i32>>,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let offset = offset.into().unwrap_or(0);
    let axis1 = axis1.into().unwrap_or(0);
    let axis2 = axis2.into().unwrap_or(1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_trace(
            result,
            a.handle,
            offset,
            axis1,
            axis2,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// Lower triangle, zeroing above the `k`th diagonal.
#[generate_macro]
#[default_device]
pub fn tril_device(
    x: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_tril(result, x.handle, k, stream.as_ref().handle)
    })
}

/// Upper triangle, zeroing below the `k`th diagonal.
#[generate_macro]
#[default_device]
pub fn triu_device(
    x: impl AsRef<Array>,
    #[optional] k: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let k = k.into().unwrap_or(0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_triu(result, x.handle, k, stream.as_ref().handle)
    })
}

/// Pad `axes` by `low_pad_size` and `high_pad_size`.
#[generate_macro]
#[default_device]
pub fn pad_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    low_pad_size: &[i32],
    high_pad_size: &[i32],
    pad_value: impl AsRef<Array>,
    mode: &str,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let pad_value = pad_value.as_ref();
    let mode_cstring = std::ffi::CString::new(mode)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_pad(
            result,
            a.handle,
            axes.as_ptr(),
            axes.len(),
            low_pad_size.as_ptr(),
            low_pad_size.len(),
            high_pad_size.as_ptr(),
            high_pad_size.len(),
            pad_value.handle,
            mode_cstring.as_ptr(),
            stream.as_ref().handle,
        )
    })
}

/// Pad every axis by `pad_width`.
#[generate_macro]
#[default_device]
pub fn pad_symmetric_device(
    a: impl AsRef<Array>,
    pad_width: i32,
    pad_value: impl AsRef<Array>,
    mode: &str,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let pad_value = pad_value.as_ref();
    let mode_cstring = std::ffi::CString::new(mode)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_pad_symmetric(
            result,
            a.handle,
            pad_width,
            pad_value.handle,
            mode_cstring.as_ptr(),
            stream.as_ref().handle,
        )
    })
}

/// A view with the given shape and strides.
///
/// The result is generally not row-contiguous; see [`Array::contiguous`](crate::Array::contiguous).
#[generate_macro]
#[default_device]
pub fn as_strided_device(
    a: impl AsRef<Array>,
    shape: &[i32],
    strides: &[i64],
    offset: usize,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_as_strided(
            result,
            a.handle,
            shape.as_ptr(),
            shape.len(),
            strides.as_ptr(),
            strides.len(),
            offset,
            stream.as_ref().handle,
        )
    })
}

/// Reinterpret the bytes as `dtype`.
#[generate_macro]
#[default_device]
pub fn view_device(
    a: impl AsRef<Array>,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_view(
            result,
            a.handle,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// The number of elements along `axes`, as an array.
#[generate_macro]
#[default_device]
pub fn number_of_elements_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    #[optional] inverted: impl Into<Option<bool>>,
    dtype: Dtype,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let inverted = inverted.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_number_of_elements(
            result,
            a.handle,
            axes.as_ptr(),
            axes.len(),
            inverted,
            mlx_rust_sys::mlx_dtype_(dtype as u32),
            stream.as_ref().handle,
        )
    })
}

/// Split at `indices` along `axis`.
#[generate_macro]
#[default_device]
pub fn split_sections_device(
    a: impl AsRef<Array>,
    indices: &[i32],
    #[optional] axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let a = a.as_ref();
    let axis = axis.into().unwrap_or(0);
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_split_sections(
            outputs.handle_mut(),
            a.handle,
            indices.as_ptr(),
            indices.len(),
            axis,
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}

/// Broadcast every input to a common shape.
#[generate_macro]
#[default_device]
pub fn broadcast_arrays_device(
    inputs: &[&Array],
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let mut inputs_vector = VectorArray::new();
    for array in inputs {
        inputs_vector.push(array)?;
    }
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_broadcast_arrays(
            outputs.handle_mut(),
            inputs_vector.handle(),
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}

/// Coordinate grids from 1-D coordinate arrays.
#[generate_macro]
#[default_device]
pub fn meshgrid_device(
    arrays: &[&Array],
    #[optional] sparse: impl Into<Option<bool>>,
    indexing: &str,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let mut arrays_vector = VectorArray::new();
    for array in arrays {
        arrays_vector.push(array)?;
    }
    let sparse = sparse.into().unwrap_or(false);
    let indexing_cstring = std::ffi::CString::new(indexing)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_meshgrid(
            outputs.handle_mut(),
            arrays_vector.handle(),
            sparse,
            indexing_cstring.as_ptr(),
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}

/// Quotient and remainder together.
#[generate_macro]
#[default_device]
pub fn divmod_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let a = a.as_ref();
    let b = b.as_ref();
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_divmod(
            outputs.handle_mut(),
            a.handle,
            b.handle,
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}
