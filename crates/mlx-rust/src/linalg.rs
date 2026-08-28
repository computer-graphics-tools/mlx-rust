//! Linear algebra, mirroring `mlx.core.linalg`.
//!
//! MLX implements most of these on the CPU only, so the defaulted forms use the
//! default CPU stream rather than the GPU one. Pass a stream explicitly with the
//! `_device` forms if you need to override that.

use mlx_rust_macros::{default_device, generate_macro};

use crate::{
    array::{Array, VectorArray},
    error::{Error, Result, check},
    stream::Stream,
};

/// Matrix inverse.
#[generate_macro]
#[default_device(cpu)]
pub fn inv_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_inv(result, a.handle, stream.as_ref().handle)
    })
}

/// Moore-Penrose pseudo-inverse.
#[generate_macro]
#[default_device(cpu)]
pub fn pinv_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_pinv(result, a.handle, stream.as_ref().handle)
    })
}

/// Inverse of a triangular matrix.
#[generate_macro]
#[default_device(cpu)]
pub fn tri_inv_device(
    a: impl AsRef<Array>,
    #[optional] upper: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let upper = upper.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_tri_inv(
            result,
            a.handle,
            upper,
            stream.as_ref().handle,
        )
    })
}

/// Cholesky factorization.
#[generate_macro]
#[default_device(cpu)]
pub fn cholesky_device(
    a: impl AsRef<Array>,
    #[optional] upper: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let upper = upper.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_cholesky(
            result,
            a.handle,
            upper,
            stream.as_ref().handle,
        )
    })
}

/// Inverse from a Cholesky factorization.
#[generate_macro]
#[default_device(cpu)]
pub fn cholesky_inv_device(
    a: impl AsRef<Array>,
    #[optional] upper: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let upper = upper.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_cholesky_inv(
            result,
            a.handle,
            upper,
            stream.as_ref().handle,
        )
    })
}

/// Cross product along `axis`.
#[generate_macro]
#[default_device(cpu)]
pub fn cross_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<i32>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let axis = axis.into().unwrap_or(-1);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_cross(
            result,
            a.handle,
            b.handle,
            axis,
            stream.as_ref().handle,
        )
    })
}

/// Solve `a x = b`.
#[generate_macro]
#[default_device(cpu)]
pub fn solve_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_solve(
            result,
            a.handle,
            b.handle,
            stream.as_ref().handle,
        )
    })
}

/// Solve `a x = b` for triangular `a`.
#[generate_macro]
#[default_device(cpu)]
pub fn solve_triangular_device(
    a: impl AsRef<Array>,
    b: impl AsRef<Array>,
    #[optional] upper: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let b = b.as_ref();
    let upper = upper.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_solve_triangular(
            result,
            a.handle,
            b.handle,
            upper,
            stream.as_ref().handle,
        )
    })
}

/// Eigenvalues of a general matrix.
#[generate_macro]
#[default_device(cpu)]
pub fn eigvals_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_eigvals(
            result,
            a.handle,
            stream.as_ref().handle,
        )
    })
}

/// Eigenvalues of a Hermitian matrix.
#[generate_macro]
#[default_device(cpu)]
pub fn eigvalsh_device(
    a: impl AsRef<Array>,
    uplo: &str,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let uplo_cstring = std::ffi::CString::new(uplo)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_eigvalsh(
            result,
            a.handle,
            uplo_cstring.as_ptr(),
            stream.as_ref().handle,
        )
    })
}

/// Eigenvalues and eigenvectors of a general matrix.
#[generate_macro]
#[default_device(cpu)]
pub fn eig_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a = a.as_ref();
    let mut first = Array::empty();
    let mut second = Array::empty();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_eig(
            &mut first.handle,
            &mut second.handle,
            a.handle,
            stream.as_ref().handle,
        )
    })?;
    Ok((first, second))
}

/// Eigenvalues and eigenvectors of a Hermitian matrix.
#[generate_macro]
#[default_device(cpu)]
pub fn eigh_device(
    a: impl AsRef<Array>,
    uplo: &str,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a = a.as_ref();
    let uplo_cstring = std::ffi::CString::new(uplo)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    let mut first = Array::empty();
    let mut second = Array::empty();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_eigh(
            &mut first.handle,
            &mut second.handle,
            a.handle,
            uplo_cstring.as_ptr(),
            stream.as_ref().handle,
        )
    })?;
    Ok((first, second))
}

/// QR factorization.
#[generate_macro]
#[default_device(cpu)]
pub fn qr_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a = a.as_ref();
    let mut first = Array::empty();
    let mut second = Array::empty();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_qr(
            &mut first.handle,
            &mut second.handle,
            a.handle,
            stream.as_ref().handle,
        )
    })?;
    Ok((first, second))
}

/// LU factorization, as `(lu, pivots)`.
#[generate_macro]
#[default_device(cpu)]
pub fn lu_factor_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<(Array, Array)> {
    let a = a.as_ref();
    let mut first = Array::empty();
    let mut second = Array::empty();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_lu_factor(
            &mut first.handle,
            &mut second.handle,
            a.handle,
            stream.as_ref().handle,
        )
    })?;
    Ok((first, second))
}

/// LU factorization as `[p, l, u]`.
#[generate_macro]
#[default_device(cpu)]
pub fn lu_device(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let a = a.as_ref();
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_lu(
            outputs.handle_mut(),
            a.handle,
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}

/// Singular value decomposition.
#[generate_macro]
#[default_device(cpu)]
pub fn svd_device(
    a: impl AsRef<Array>,
    #[optional] compute_uv: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Vec<Array>> {
    let a = a.as_ref();
    let compute_uv = compute_uv.into().unwrap_or(true);
    let mut outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_linalg_svd(
            outputs.handle_mut(),
            a.handle,
            compute_uv,
            stream.as_ref().handle,
        )
    })?;
    (0..outputs.len()).map(|index| outputs.get(index)).collect()
}

/// Vector or matrix norm of order `ord`.
#[generate_macro]
#[default_device(cpu)]
pub fn norm_device<'a>(
    a: impl AsRef<Array>,
    ord: f64,
    #[optional] axis: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into();
    let keepdims = keepdims.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_norm(
            result,
            a.handle,
            ord,
            axis.map_or(::core::ptr::null(), <[i32]>::as_ptr),
            axis.map_or(0, <[i32]>::len),
            keepdims,
            stream.as_ref().handle,
        )
    })
}

/// Matrix norm named by `ord`, such as `"fro"`.
#[generate_macro]
#[default_device(cpu)]
pub fn norm_matrix_device<'a>(
    a: impl AsRef<Array>,
    ord: &str,
    #[optional] axis: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into();
    let keepdims = keepdims.into().unwrap_or(false);
    let ord_cstring = std::ffi::CString::new(ord)
        .map_err(|_| Error::Invalid("argument contains a NUL".into()))?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_norm_matrix(
            result,
            a.handle,
            ord_cstring.as_ptr(),
            axis.map_or(::core::ptr::null(), <[i32]>::as_ptr),
            axis.map_or(0, <[i32]>::len),
            keepdims,
            stream.as_ref().handle,
        )
    })
}

/// L2 norm.
#[generate_macro]
#[default_device(cpu)]
pub fn norm_l2_device<'a>(
    a: impl AsRef<Array>,
    #[optional] axis: impl Into<Option<&'a [i32]>>,
    #[optional] keepdims: impl Into<Option<bool>>,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let a = a.as_ref();
    let axis = axis.into();
    let keepdims = keepdims.into().unwrap_or(false);
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_linalg_norm_l2(
            result,
            a.handle,
            axis.map_or(::core::ptr::null(), <[i32]>::as_ptr),
            axis.map_or(0, <[i32]>::len),
            keepdims,
            stream.as_ref().handle,
        )
    })
}
