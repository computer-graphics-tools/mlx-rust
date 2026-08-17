//! Safe Rust bindings to [MLX](https://github.com/ml-explore/mlx), scoped to
//! quantized-GEMM benchmarking.
//!
//! This is deliberately not a general framework binding: there are no neural
//! network modules, optimizers, or autodiff. It exposes just enough to build
//! quantized weights, run `quantized_matmul` / `gather_qmm` against them, and
//! measure the result honestly.
//!
//! ```no_run
//! use mlx::{Array, ops};
//!
//! let weights = Array::from_slice(&vec![0.1f32; 512 * 512], &[512, 512])?;
//! let (quantized_weights, scales, biases) = ops::quantize(&weights, None)?;
//!
//! let input = Array::from_slice(&vec![0.5f32; 512], &[1, 512])?;
//! let output = ops::quantized_matmul(
//!     &input, &quantized_weights, &scales, biases.as_ref(), None, None,
//! )?;
//! println!("{:?}", output.to_vec_f32()?.len());
//! # Ok::<(), mlx::Error>(())
//! ```
//!
//! Each op has a `_device` twin taking an explicit [`Stream`], and every optional
//! argument accepts a value in place of `None`:
//!
//! ```no_run
//! use mlx::{Array, Stream, ops::{self, QuantConfig}};
//!
//! let stream = Stream::cpu();
//! let config = QuantConfig::affine(32, 8);
//! let weights = Array::from_slice(&vec![0.1f32; 512 * 512], &[512, 512])?;
//! let (quantized_weights, scales, biases) =
//!     ops::quantize_device(&weights, config, &stream)?;
//! # Ok::<(), mlx::Error>(())
//! ```
//!
//! ## Threading
//!
//! [`Array`] and [`Stream`] are neither `Send` nor `Sync`. MLX arrays are not
//! thread-safe and the default streams are process-global; use one thread.
//!
//! ## Errors
//!
//! mlx-c's default error handler calls `exit(-1)`. This crate installs its own on
//! first use so failures surface as [`Error`]. Nothing needs to opt in.
//!
//! ## Reading results back
//!
//! [`Array::to_vec`] and [`Array::to_vec_f32`] copy a row-major run of elements
//! and error on any other layout. [`ops::slice`] returns a view, so a strided or
//! reversed slice needs [`Array::contiguous`] first; the quantized ops already
//! return contiguous arrays.

/// Owned MLX arrays and host transfers.
pub mod array;
/// Element types and the mapping to MLX's `mlx_dtype`.
pub mod dtype;
/// Error type and mlx-c error-handler installation.
pub mod error;
/// Metal backend capability probing, capture, and allocator statistics.
pub mod metal;
/// The quantized and elementwise operations this crate exposes.
pub mod ops;
/// MLX execution streams.
pub mod stream;

pub use array::{Array, eval_all};
pub use dtype::{Dtype, Element};
pub use error::{Error, Result};
pub use ops::{QuantConfig, QuantMode};
pub use stream::Stream;

/// An owned `mlx_string`, freed on drop.
struct MlxString {
    handle: mlx_rust_sys::mlx_string,
}

impl MlxString {
    fn new() -> Self {
        error::install();
        MlxString {
            handle: unsafe { mlx_rust_sys::mlx_string_new() },
        }
    }

    fn handle_mut(&mut self) -> &mut mlx_rust_sys::mlx_string {
        &mut self.handle
    }

    fn to_string_lossy(&self) -> String {
        let text_ptr = unsafe { mlx_rust_sys::mlx_string_data(self.handle) };
        if text_ptr.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(text_ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_string_free(self.handle) };
        }
    }
}

/// The version of the MLX this crate is linked against.
pub fn mlx_version() -> Result<String> {
    let mut version = MlxString::new();
    error::check(|| unsafe {
        mlx_rust_sys::mlx_version(version.handle_mut())
    })?;
    Ok(version.to_string_lossy())
}
