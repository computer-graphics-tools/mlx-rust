//! Slicing.

use mlx_rust_macros::default_device;

use crate::{
    array::Array,
    error::{Error, Result},
    stream::Stream,
};

/// Strided slice, `a[start..stop:strides]` per axis.
///
/// The result is a view sharing `a`'s buffer, so unless every stride is 1 it is
/// not row-contiguous and needs [`Array::contiguous`] before a host read.
#[default_device]
pub fn slice_device(
    a: impl AsRef<Array>,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let array = a.as_ref();
    if start.len() != stop.len() || start.len() != strides.len() {
        return Err(Error::Invalid(
            "slice start/stop/strides must have equal length".into(),
        ));
    }
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_slice(
            result,
            array.handle,
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
