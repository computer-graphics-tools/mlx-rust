//! Indexing: take, gather, scatter and slice updates.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    array::{Array, VectorArray},
    error::Result,
    stream::Stream,
};

/// Take elements at flat `indices`.
#[generate_macro]
#[default_device]
pub fn take_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_take(
            result,
            a.handle,
            indices.handle,
            stream.as_ref().handle,
        )
    })
}

/// Take elements at `indices` along `axis`.
#[generate_macro]
#[default_device]
pub fn take_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_take_axis(
            result,
            a.handle,
            indices.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Take elements matching `indices` elementwise along `axis`.
#[generate_macro]
#[default_device]
pub fn take_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_take_along_axis(
            result,
            a.handle,
            indices.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Put `values` at `indices` along `axis`.
#[generate_macro]
#[default_device]
pub fn put_along_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    values: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let values = values.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_put_along_axis(
            result,
            a.handle,
            indices.handle,
            values.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Gather slices of `slice_sizes` at `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn gather_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    axes: &[i32],
    slice_sizes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_gather(
            result,
            a.handle,
            indices_vector.handle(),
            axes.as_ptr(),
            axes.len(),
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Gather slices of `slice_sizes` at `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn gather_single_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    axis: i32,
    slice_sizes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_gather_single(
            result,
            a.handle,
            indices.handle,
            axis,
            slice_sizes.as_ptr(),
            slice_sizes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter `updates` to `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn scatter_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    updates: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter(
            result,
            a.handle,
            indices_vector.handle(),
            updates.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter-add `updates` to `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn scatter_add_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    updates: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_add(
            result,
            a.handle,
            indices_vector.handle(),
            updates.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter-maximum `updates` to `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn scatter_max_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    updates: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_max(
            result,
            a.handle,
            indices_vector.handle(),
            updates.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter-minimum `updates` to `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn scatter_min_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    updates: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_min(
            result,
            a.handle,
            indices_vector.handle(),
            updates.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter-product `updates` to `indices` along `axes`.
#[generate_macro]
#[default_device]
pub fn scatter_prod_device(
    a: impl AsRef<Array>,
    indices: &[&Array],
    updates: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mut indices_vector = VectorArray::new();
    for array in indices {
        indices_vector.push(array)?;
    }
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_prod(
            result,
            a.handle,
            indices_vector.handle(),
            updates.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter `updates` to `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn scatter_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_single(
            result,
            a.handle,
            indices.handle,
            updates.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Scatter-add `values` to `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn scatter_add_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    values: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let values = values.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_add_axis(
            result,
            a.handle,
            indices.handle,
            values.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Write `src` into the positions where `mask` is true.
#[generate_macro]
#[default_device]
pub fn masked_scatter_device(
    a: impl AsRef<Array>,
    mask: impl AsRef<Array>,
    src: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let mask = mask.as_ref();
    let src = src.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_masked_scatter(
            result,
            a.handle,
            mask.handle,
            src.handle,
            stream.as_ref().handle,
        )
    })
}

/// Slice of `slice_size` beginning at the runtime `start` along `axes`.
#[generate_macro]
#[default_device]
pub fn slice_dynamic_device(
    a: impl AsRef<Array>,
    start: impl AsRef<Array>,
    axes: &[i32],
    slice_size: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let start = start.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_dynamic(
            result,
            a.handle,
            start.handle,
            axes.as_ptr(),
            axes.len(),
            slice_size.as_ptr(),
            slice_size.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with the strided slice replaced by `update`.
#[generate_macro]
#[default_device]
pub fn slice_update_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update(
            result,
            src.handle,
            update.handle,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with `update` added into the strided slice.
#[generate_macro]
#[default_device]
pub fn slice_update_add_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update_add(
            result,
            src.handle,
            update.handle,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with the strided slice replaced by the elementwise maximum.
#[generate_macro]
#[default_device]
pub fn slice_update_max_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update_max(
            result,
            src.handle,
            update.handle,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with the strided slice replaced by the elementwise minimum.
#[generate_macro]
#[default_device]
pub fn slice_update_min_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update_min(
            result,
            src.handle,
            update.handle,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with the strided slice multiplied by `update`.
#[generate_macro]
#[default_device]
pub fn slice_update_prod_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update_prod(
            result,
            src.handle,
            update.handle,
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            strides.as_ptr(),
            strides.len(),
            stream.as_ref().handle,
        )
    })
}

/// `src` with the slice at the runtime `start` replaced by `update`.
#[generate_macro]
#[default_device]
pub fn slice_update_dynamic_device(
    src: impl AsRef<Array>,
    update: impl AsRef<Array>,
    start: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let src = src.as_ref();
    let update = update.as_ref();
    let start = start.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice_update_dynamic(
            result,
            src.handle,
            update.handle,
            start.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Scatter-max `updates` to `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn scatter_max_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_max_single(
            result,
            a.handle,
            indices.handle,
            updates.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Scatter-min `updates` to `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn scatter_min_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_min_single(
            result,
            a.handle,
            indices.handle,
            updates.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Scatter-prod `updates` to `indices` along one `axis`.
#[generate_macro]
#[default_device]
pub fn scatter_prod_axis_device(
    a: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    updates: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let indices = indices.as_ref();
    let updates = updates.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_scatter_prod_single(
            result,
            a.handle,
            indices.handle,
            updates.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}
