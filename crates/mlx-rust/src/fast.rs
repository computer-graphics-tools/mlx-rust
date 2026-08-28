//! Fused kernels, mirroring `mlx.core.fast`.
#![expect(clippy::too_many_arguments, reason = "mirrors mlx-c")]

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    array::{Array, null_array},
    error::{Error, Result},
    ops::optional_float,
    stream::Stream,
};

/// Fused layer normalization.
#[generate_macro]
#[default_device]
pub fn layer_norm_device<'a>(
    x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    eps: f32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let weight = weight.into();
    let bias = bias.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fast_layer_norm(
            result,
            x.handle,
            weight.map(|array| array.handle).unwrap_or_else(null_array),
            bias.map(|array| array.handle).unwrap_or_else(null_array),
            eps,
            stream.as_ref().handle,
        )
    })
}

/// Fused RMS normalization.
#[generate_macro]
#[default_device]
pub fn rms_norm_device<'a>(
    x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    eps: f32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let weight = weight.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fast_rms_norm(
            result,
            x.handle,
            weight.map(|array| array.handle).unwrap_or_else(null_array),
            eps,
            stream.as_ref().handle,
        )
    })
}

/// Rotary positional encoding.
#[generate_macro]
#[default_device]
pub fn rope_device<'a>(
    x: impl AsRef<Array>,
    dims: i32,
    traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    scale: f32,
    offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let base = optional_float(base.into());
    let freqs = freqs.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fast_rope(
            result,
            x.handle,
            dims,
            traditional,
            base,
            scale,
            offset,
            freqs.map(|array| array.handle).unwrap_or_else(null_array),
            stream.as_ref().handle,
        )
    })
}

/// Rotary positional encoding with a runtime `offset`.
#[generate_macro]
#[default_device]
pub fn rope_dynamic_device<'a>(
    x: impl AsRef<Array>,
    dims: i32,
    traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    scale: f32,
    offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let x = x.as_ref();
    let base = optional_float(base.into());
    let offset = offset.as_ref();
    let freqs = freqs.into();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fast_rope_dynamic(
            result,
            x.handle,
            dims,
            traditional,
            base,
            scale,
            offset.handle,
            freqs.map(|array| array.handle).unwrap_or_else(null_array),
            stream.as_ref().handle,
        )
    })
}

/// Fused scaled dot-product attention.
///
/// `mask_mode` is `""`, `"causal"` or `"array"`.
#[generate_macro]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    mask_mode: &str,
    #[optional] mask_arr: impl Into<Option<&'a Array>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let queries = queries.as_ref();
    let keys = keys.as_ref();
    let values = values.as_ref();
    let mask_arr = mask_arr.into();
    let sinks = sinks.into();
    let mask_mode_cstring = std::ffi::CString::new(mask_mode)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fast_scaled_dot_product_attention(
            result,
            queries.handle,
            keys.handle,
            values.handle,
            scale,
            mask_mode_cstring.as_ptr(),
            mask_arr.map(|array| array.handle).unwrap_or_else(null_array),
            sinks.map(|array| array.handle).unwrap_or_else(null_array),
            stream.as_ref().handle,
        )
    })
}
