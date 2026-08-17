use std::ffi::CStr;

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    array::{Array, VectorArray, null_array},
    dtype::Dtype,
    error::{Error, Result, check},
    stream::Stream,
};

/// Quantization scheme.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum QuantMode {
    /// `w = scale * q + bias`. The only mode that takes biases.
    #[default]
    Affine,
    /// OCP MX 4-bit float.
    Mxfp4,
    /// OCP MX 8-bit float.
    Mxfp8,
    /// NVIDIA 4-bit float.
    Nvfp4,
}

impl QuantMode {
    /// The mode string MLX matches on.
    pub(crate) fn as_c_str(self) -> &'static CStr {
        match self {
            QuantMode::Affine => c"affine",
            QuantMode::Mxfp4 => c"mxfp4",
            QuantMode::Mxfp8 => c"mxfp8",
            QuantMode::Nvfp4 => c"nvfp4",
        }
    }

    /// The mode string as UTF-8.
    pub fn as_str(self) -> &'static str {
        match self {
            QuantMode::Affine => "affine",
            QuantMode::Mxfp4 => "mxfp4",
            QuantMode::Mxfp8 => "mxfp8",
            QuantMode::Nvfp4 => "nvfp4",
        }
    }

    /// Whether this mode takes biases. Only [`Affine`](Self::Affine) does.
    pub fn uses_biases(self) -> bool {
        matches!(self, QuantMode::Affine)
    }

    /// The `(group_size, bits)` this mode defaults to. Fixed by the format for
    /// every mode except [`Affine`](Self::Affine).
    pub fn default_params(self) -> (i32, i32) {
        match self {
            QuantMode::Affine => (64, 4),
            QuantMode::Mxfp4 => (32, 4),
            QuantMode::Mxfp8 => (32, 8),
            QuantMode::Nvfp4 => (16, 4),
        }
    }
}

/// Parameters shared by the quantized ops. [`Default`] is `affine(64, 4)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuantConfig {
    /// Elements sharing one scale, along the last axis.
    pub group_size: i32,
    /// Bit width of each quantized code.
    pub bits: i32,
    /// The quantization scheme.
    pub mode: QuantMode,
}

impl Default for QuantConfig {
    fn default() -> Self {
        QuantConfig::with_defaults(QuantMode::Affine)
    }
}

impl QuantConfig {
    /// Affine quantization with the given group size and bit width.
    pub fn affine(
        group_size: i32,
        bits: i32,
    ) -> Self {
        QuantConfig {
            group_size,
            bits,
            mode: QuantMode::Affine,
        }
    }

    /// A config for `mode` using [`QuantMode::default_params`].
    pub fn with_defaults(mode: QuantMode) -> Self {
        let (group_size, bits) = mode.default_params();
        QuantConfig {
            group_size,
            bits,
            mode,
        }
    }

    /// Check this config against `biases`.
    ///
    /// # Errors
    ///
    /// If `group_size` or `bits` is not positive, if `biases` is present for a
    /// mode that takes none (or absent for [`Affine`](QuantMode::Affine)), or if
    /// a non-affine mode's layout differs from [`QuantMode::default_params`].
    pub fn validate(
        &self,
        biases: Option<&Array>,
    ) -> Result<()> {
        if self.group_size <= 0 || self.bits <= 0 {
            return Err(Error::Invalid(format!(
                "{} quantization needs positive group_size and bits, got {} and {}",
                self.mode.as_str(),
                self.group_size,
                self.bits
            )));
        }
        match (self.mode.uses_biases(), biases.is_some()) {
            (true, false) => {
                return Err(Error::Invalid(format!(
                    "{} quantization requires biases",
                    self.mode.as_str()
                )));
            },
            (false, true) => {
                return Err(Error::Invalid(format!(
                    "{} quantization must not be given biases",
                    self.mode.as_str()
                )));
            },
            _ => {},
        }
        if !self.mode.uses_biases() {
            let (group_size, bits) = self.mode.default_params();
            if (self.group_size, self.bits) != (group_size, bits) {
                return Err(Error::Invalid(format!(
                    "{} quantization is fixed at group_size {group_size} and \
                     bits {bits}, got {} and {}",
                    self.mode.as_str(),
                    self.group_size,
                    self.bits
                )));
            }
        }
        Ok(())
    }

    fn group_size_opt(&self) -> mlx_rust_sys::mlx_optional_int {
        mlx_rust_sys::mlx_optional_int {
            value: self.group_size,
            has_value: true,
        }
    }

    fn bits_opt(&self) -> mlx_rust_sys::mlx_optional_int {
        mlx_rust_sys::mlx_optional_int {
            value: self.bits,
            has_value: true,
        }
    }
}

fn opt_dtype(dtype: Option<Dtype>) -> mlx_rust_sys::mlx_optional_dtype {
    mlx_rust_sys::mlx_optional_dtype {
        value: mlx_rust_sys::mlx_dtype_(dtype.unwrap_or(Dtype::Float32) as u32),
        has_value: dtype.is_some(),
    }
}

/// `x @ dequantize(w).T` when `transpose`, else `x @ dequantize(w)`.
///
/// # Errors
///
/// If `config` and `biases` disagree; see [`QuantConfig::validate`].
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] config: impl Into<Option<QuantConfig>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (input, quantized_weights, scales) =
        (x.as_ref(), w.as_ref(), scales.as_ref());
    let biases = biases.into();
    let transpose = transpose.into().unwrap_or(true);
    let config = config.into().unwrap_or_default();
    config.validate(biases)?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_quantized_matmul(
            result,
            input.handle,
            quantized_weights.handle,
            scales.handle,
            biases.map(|biases| biases.handle).unwrap_or_else(null_array),
            transpose,
            config.group_size_opt(),
            config.bits_opt(),
            config.mode.as_c_str().as_ptr(),
            stream.as_ref().handle,
        )
    })
}

/// Quantize `w`, returning `(w_q, scales, biases)`.
///
/// `biases` is `None` for modes that do not produce one.
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] config: impl Into<Option<QuantConfig>>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Option<Array>)> {
    let weights = w.as_ref();
    let config = config.into().unwrap_or_default();
    // Biases are an output here, so stand in whatever this mode will produce.
    config.validate(if config.mode.uses_biases() {
        Some(weights)
    } else {
        None
    })?;

    let mut results = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_quantize(
            results.handle_mut(),
            weights.handle,
            config.group_size_opt(),
            config.bits_opt(),
            config.mode.as_c_str().as_ptr(),
            null_array(),
            stream.as_ref().handle,
        )
    })?;

    let returned = results.len();
    if returned < 2 {
        return Err(Error::Invalid(format!(
            "mlx_quantize returned {returned} arrays, expected at least 2"
        )));
    }
    let quantized_weights = results.get(0)?;
    let scales = results.get(1)?;
    let biases = if returned > 2 {
        Some(results.get(2)?)
    } else {
        None
    };
    Ok((quantized_weights, scales, biases))
}

/// Reconstruct full-precision weights from a quantized triple.
///
/// For [`Affine`](QuantMode::Affine) this is `w = scale * q + bias` over groups
/// of `group_size` along the last axis, with codes packed low-order-first into
/// each `uint32`.
#[generate_macro]
#[default_device]
pub fn dequantize_device<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] config: impl Into<Option<QuantConfig>>,
    #[optional] dtype: impl Into<Option<Dtype>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (quantized_weights, scales) = (w.as_ref(), scales.as_ref());
    let biases = biases.into();
    let config = config.into().unwrap_or_default();
    let dtype = dtype.into();
    config.validate(biases)?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_dequantize(
            result,
            quantized_weights.handle,
            scales.handle,
            biases.map(|biases| biases.handle).unwrap_or_else(null_array),
            config.group_size_opt(),
            config.bits_opt(),
            config.mode.as_c_str().as_ptr(),
            null_array(),
            opt_dtype(dtype),
            stream.as_ref().handle,
        )
    })
}

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

/// Quantized matmul with per-batch gathering of `x` and `w`.
#[expect(clippy::too_many_arguments, reason = "mirrors mlx_gather_qmm")]
#[generate_macro]
#[default_device]
pub fn gather_qmm_device<'a, 'l, 'r>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] lhs_indices: impl Into<Option<&'l Array>>,
    #[optional] rhs_indices: impl Into<Option<&'r Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] config: impl Into<Option<QuantConfig>>,
    #[optional] sorted_indices: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let (input, quantized_weights, scales) =
        (x.as_ref(), w.as_ref(), scales.as_ref());
    let biases = biases.into();
    let lhs_indices = lhs_indices.into();
    let rhs_indices = rhs_indices.into();
    let transpose = transpose.into().unwrap_or(true);
    let config = config.into().unwrap_or_default();
    let sorted_indices = sorted_indices.into().unwrap_or(false);
    config.validate(biases)?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_gather_qmm(
            result,
            input.handle,
            quantized_weights.handle,
            scales.handle,
            biases.map(|biases| biases.handle).unwrap_or_else(null_array),
            lhs_indices
                .map(|indices| indices.handle)
                .unwrap_or_else(null_array),
            rhs_indices
                .map(|indices| indices.handle)
                .unwrap_or_else(null_array),
            transpose,
            config.group_size_opt(),
            config.bits_opt(),
            config.mode.as_c_str().as_ptr(),
            sorted_indices,
            stream.as_ref().handle,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `as_str` and `as_c_str` are independent matches, so pin them together.
    /// `as_c_str` is `pub(crate)`, so this is the only place it is reachable.
    #[test]
    fn mode_strings_agree() {
        for mode in [
            QuantMode::Affine,
            QuantMode::Mxfp4,
            QuantMode::Mxfp8,
            QuantMode::Nvfp4,
        ] {
            assert_eq!(mode.as_c_str().to_str().unwrap(), mode.as_str());
        }
    }
}
