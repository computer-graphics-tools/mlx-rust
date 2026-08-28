//! Sorting, partitioning and selection.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{array::Array, error::Result, stream::Stream};

/// Sort along `axis`.
#[generate_macro]
#[default_device]
pub fn sort_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_sort_axis(
            result,
            a.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Indices that sort along `axis`.
#[generate_macro]
#[default_device]
pub fn argsort_axis_device(
    a: impl AsRef<Array>,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_argsort_axis(
            result,
            a.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Partition so the `kth` element is in its sorted position.
#[generate_macro]
#[default_device]
pub fn partition_device(
    a: impl AsRef<Array>,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_partition(
            result,
            a.handle,
            kth,
            stream.as_ref().handle,
        )
    })
}

/// Partition along `axis`.
#[generate_macro]
#[default_device]
pub fn partition_axis_device(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_partition_axis(
            result,
            a.handle,
            kth,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Indices that partition at `kth`.
#[generate_macro]
#[default_device]
pub fn argpartition_device(
    a: impl AsRef<Array>,
    kth: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_argpartition(
            result,
            a.handle,
            kth,
            stream.as_ref().handle,
        )
    })
}

/// Indices that partition at `kth` along `axis`.
#[generate_macro]
#[default_device]
pub fn argpartition_axis_device(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_argpartition_axis(
            result,
            a.handle,
            kth,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// The `k` largest elements.
#[generate_macro]
#[default_device]
pub fn topk_device(
    a: impl AsRef<Array>,
    k: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_topk(result, a.handle, k, stream.as_ref().handle)
    })
}

/// The `k` largest elements along `axis`.
#[generate_macro]
#[default_device]
pub fn topk_axis_device(
    a: impl AsRef<Array>,
    k: i32,
    axis: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_topk_axis(
            result,
            a.handle,
            k,
            axis,
            stream.as_ref().handle,
        )
    })
}
