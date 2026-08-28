//! Shared shapes for the ops whose mlx-c signature is uniform.
//!
//! Wrapping ~600 functions one at a time would be almost entirely repetition, so
//! the two shapes that recur most get a declarative macro. Each op still names
//! itself and its mlx-c symbol at the call site, and carries its own docs.

/// `int mlx_op(mlx_array* res, const mlx_array a, const mlx_stream s)`
macro_rules! unary_ops {
    ($($(#[$attr:meta])* $device_name:ident => $ffi:ident,)*) => {$(
        $(#[$attr])*
        #[::mlx_rust_macros::default_device]
        pub fn $device_name(
            a: impl AsRef<$crate::Array>,
            stream: impl AsRef<$crate::Stream>,
        ) -> $crate::Result<$crate::Array> {
            $crate::Array::try_from_op(|result| unsafe {
                mlx_rust_sys::$ffi(
                    result,
                    a.as_ref().handle,
                    stream.as_ref().handle,
                )
            })
        }
    )*};
}

/// `int mlx_op(mlx_array* res, const mlx_array a, const mlx_array b,
/// const mlx_stream s)`
macro_rules! binary_ops {
    ($($(#[$attr:meta])* $device_name:ident => $ffi:ident,)*) => {$(
        $(#[$attr])*
        #[::mlx_rust_macros::default_device]
        pub fn $device_name(
            a: impl AsRef<$crate::Array>,
            b: impl AsRef<$crate::Array>,
            stream: impl AsRef<$crate::Stream>,
        ) -> $crate::Result<$crate::Array> {
            $crate::Array::try_from_op(|result| unsafe {
                mlx_rust_sys::$ffi(
                    result,
                    a.as_ref().handle,
                    b.as_ref().handle,
                    stream.as_ref().handle,
                )
            })
        }
    )*};
}

pub(crate) use binary_ops;
pub(crate) use unary_ops;
