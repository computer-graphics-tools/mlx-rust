//! Fast Fourier transforms, mirroring `mlx.core.fft`.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{array::Array, error::Result, stream::Stream};

/// How a transform is normalized.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum FftNorm {
    /// No scaling on the forward transform; `1/n` on the inverse.
    #[default]
    Backward = 0,
    /// `1/sqrt(n)` on both.
    Ortho = 1,
    /// `1/n` on the forward transform; none on the inverse.
    Forward = 2,
}

/// Discrete Fourier transform along one `axis`.
#[generate_macro]
#[default_device]
pub fn fft_device(
    a: impl AsRef<Array>,
    n: i32,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into().unwrap_or(-1);
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_fft(
            result,
            a.handle,
            n,
            axis,
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Discrete Fourier transform over `axes`.
#[generate_macro]
#[default_device]
pub fn fftn_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_fftn(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Two-dimensional discrete fourier transform.
#[generate_macro]
#[default_device]
pub fn fft2_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_fft2(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Inverse DFT along one `axis`.
#[generate_macro]
#[default_device]
pub fn ifft_device(
    a: impl AsRef<Array>,
    n: i32,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into().unwrap_or(-1);
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_ifft(
            result,
            a.handle,
            n,
            axis,
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Inverse DFT over `axes`.
#[generate_macro]
#[default_device]
pub fn ifftn_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_ifftn(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Two-dimensional inverse dft.
#[generate_macro]
#[default_device]
pub fn ifft2_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_ifft2(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Real-input DFT along one `axis`.
#[generate_macro]
#[default_device]
pub fn rfft_device(
    a: impl AsRef<Array>,
    n: i32,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into().unwrap_or(-1);
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_rfft(
            result,
            a.handle,
            n,
            axis,
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Real-input DFT over `axes`.
#[generate_macro]
#[default_device]
pub fn rfftn_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_rfftn(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Two-dimensional real-input dft.
#[generate_macro]
#[default_device]
pub fn rfft2_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_rfft2(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Inverse real-output DFT along one `axis`.
#[generate_macro]
#[default_device]
pub fn irfft_device(
    a: impl AsRef<Array>,
    n: i32,
    #[optional] axis: impl Into<Option<i32>>,
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into().unwrap_or(-1);
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_irfft(
            result,
            a.handle,
            n,
            axis,
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Inverse real-output DFT over `axes`.
#[generate_macro]
#[default_device]
pub fn irfftn_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_irfftn(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Two-dimensional inverse real-output dft.
#[generate_macro]
#[default_device]
pub fn irfft2_device(
    a: impl AsRef<Array>,
    n: &[i32],
    axes: &[i32],
    #[optional] norm: impl Into<Option<FftNorm>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let norm = norm.into().unwrap_or(FftNorm::Backward);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_irfft2(
            result,
            a.handle,
            n.as_ptr(),
            n.len(),
            axes.as_ptr(),
            axes.len(),
            mlx_rust_sys::mlx_fft_norm_(norm as u32),
            stream.as_ref().handle,
        )
    })
}

/// Shift the zero frequency to the centre.
#[generate_macro]
#[default_device]
pub fn fftshift_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_fftshift(
            result,
            a.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Undo [`fftshift`].
#[generate_macro]
#[default_device]
pub fn ifftshift_device(
    a: impl AsRef<Array>,
    axes: &[i32],
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_ifftshift(
            result,
            a.handle,
            axes.as_ptr(),
            axes.len(),
            stream.as_ref().handle,
        )
    })
}

/// Sample frequencies for a length-`n` transform.
#[generate_macro]
#[default_device]
pub fn fftfreq_device(
    n: i32,
    #[optional] d: impl Into<Option<f64>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let d = d.into().unwrap_or(1.0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_fftfreq(result, n, d, stream.as_ref().handle)
    })
}

/// Sample frequencies for a real transform.
#[generate_macro]
#[default_device]
pub fn rfftfreq_device(
    n: i32,
    #[optional] d: impl Into<Option<f64>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let d = d.into().unwrap_or(1.0);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_fft_rfftfreq(result, n, d, stream.as_ref().handle)
    })
}
