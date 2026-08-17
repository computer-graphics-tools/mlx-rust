use std::ffi::c_void;

use mlx_rust_macros::default_device;

use crate::{
    dtype::{Dtype, Element},
    error::{Error, Result, check, install},
    stream::Stream,
};

/// An MLX array.
///
/// Owns an `mlx_array` handle and frees it on drop. Neither `Send` nor `Sync`:
/// MLX arrays are not thread-safe and the default streams are process-global.
pub struct Array {
    pub(crate) handle: mlx_rust_sys::mlx_array,
}

/// A null handle, which mlx-c reads as "argument not provided".
pub(crate) fn null_array() -> mlx_rust_sys::mlx_array {
    mlx_rust_sys::mlx_array_ {
        ctx: std::ptr::null_mut(),
    }
}

/// Number of elements a shape describes, rejecting negative and overflowing
/// dimensions.
fn element_count(shape: &[i32]) -> Result<usize> {
    let mut total: usize = 1;
    for (axis, &dim) in shape.iter().enumerate() {
        let dim = usize::try_from(dim).map_err(|_| {
            Error::Invalid(format!(
                "shape {shape:?} has a negative dimension {dim} at axis {axis}"
            ))
        })?;
        total = total.checked_mul(dim).ok_or_else(|| {
            Error::Invalid(format!(
                "shape {shape:?} overflows usize at axis {axis}"
            ))
        })?;
    }
    Ok(total)
}

impl Array {
    fn from_handle(handle: mlx_rust_sys::mlx_array) -> Self {
        Array {
            handle,
        }
    }

    /// A fresh handle for use as an out-parameter.
    pub(crate) fn empty() -> Self {
        install();
        Array::from_handle(unsafe { mlx_rust_sys::mlx_array_new() })
    }

    /// Run an mlx-c op that writes its result into an out-parameter.
    pub(crate) fn try_from_op(
        operation: impl FnOnce(&mut mlx_rust_sys::mlx_array) -> i32
    ) -> Result<Array> {
        let mut output = Array::empty();
        check(|| operation(&mut output.handle))?;
        Ok(output)
    }

    /// Upload `data` as an array of the given shape.
    ///
    /// The buffer is copied; `data` is not borrowed past this call.
    pub fn from_slice<Value: Element>(
        data: &[Value],
        shape: &[i32],
    ) -> Result<Self> {
        install();
        let ndim = i32::try_from(shape.len()).map_err(|_| {
            Error::Invalid(format!("shape has {} axes", shape.len()))
        })?;
        let expected = element_count(shape)?;
        if expected != data.len() {
            return Err(Error::Invalid(format!(
                "shape {shape:?} implies {expected} elements but {} were given",
                data.len()
            )));
        }
        crate::error::clear_last_error();
        let handle = unsafe {
            mlx_rust_sys::mlx_array_new_data(
                data.as_ptr().cast::<c_void>(),
                shape.as_ptr(),
                ndim,
                mlx_rust_sys::mlx_dtype_(Value::DTYPE as u32),
            )
        };
        if handle.ctx.is_null() {
            return Err(Error::Mlx {
                message: crate::error::take_last_error().unwrap_or_else(|| {
                    format!(
                        "mlx_array_new_data returned null for shape {shape:?} \
                         dtype {:?} ({} elements)",
                        Value::DTYPE,
                        data.len()
                    )
                }),
            });
        }
        Ok(Array::from_handle(handle))
    }

    /// This array's element type.
    pub fn dtype(&self) -> Result<Dtype> {
        Dtype::from_raw(unsafe { mlx_rust_sys::mlx_array_dtype(self.handle) }.0)
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        unsafe { mlx_rust_sys::mlx_array_size(self.handle) }
    }

    /// Number of axes.
    pub fn ndim(&self) -> usize {
        unsafe { mlx_rust_sys::mlx_array_ndim(self.handle) }
    }

    /// Length of each axis.
    ///
    /// Borrowed because MLX fixes an array's shape at construction. `strides`
    /// cannot be: `copy_shared_buffer` reassigns those during eval.
    pub fn shape(&self) -> &[i32] {
        let axis_count = self.ndim();
        let shape_ptr = unsafe { mlx_rust_sys::mlx_array_shape(self.handle) };
        if shape_ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(shape_ptr, axis_count) }
    }

    /// Strides in elements, one per axis.
    ///
    /// Read as `i64`: mlx-c declares `const size_t*` but casts MLX's signed
    /// strides, which are negative for a reversed slice.
    fn strides(&self) -> Vec<i64> {
        let axis_count = self.ndim();
        let strides_ptr =
            unsafe { mlx_rust_sys::mlx_array_strides(self.handle) };
        if strides_ptr.is_null() {
            return Vec::new();
        }
        unsafe {
            std::slice::from_raw_parts(strides_ptr.cast::<i64>(), axis_count)
        }
        .to_vec()
    }

    /// Whether the elements are laid out contiguously in row-major order.
    fn is_row_contiguous(
        shape: &[i32],
        strides: &[i64],
    ) -> bool {
        if shape.len() != strides.len() {
            return false;
        }
        let mut expected_stride: i64 = 1;
        for (&axis_length, &stride) in shape.iter().zip(strides).rev() {
            if stride != expected_stride && axis_length != 1 {
                return false;
            }
            expected_stride *= i64::from(axis_length);
        }
        true
    }

    /// A row-contiguous copy, needed before reading a [`slice`](crate::ops::slice)
    /// on the host.
    #[default_device]
    pub fn contiguous_device(
        &self,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_rust_sys::mlx_contiguous(
                res,
                self.handle,
                false,
                stream.as_ref().handle,
            )
        })
    }

    fn require_row_contiguous(&self) -> Result<()> {
        let (shape, strides) = (self.shape(), self.strides());
        if Self::is_row_contiguous(shape, &strides) {
            return Ok(());
        }
        Err(Error::Invalid(format!(
            "array with shape {shape:?} and strides {strides:?} is not \
             row-contiguous; call `contiguous()` before reading it on the host"
        )))
    }

    /// Force evaluation of this array's graph.
    pub fn eval(&self) -> Result<()> {
        check(|| unsafe { mlx_rust_sys::mlx_array_eval(self.handle) })
    }

    /// Convert to `dtype`, producing a new array.
    #[default_device]
    pub fn astype_device(
        &self,
        dtype: Dtype,
        stream: impl AsRef<Stream>,
    ) -> Result<Array> {
        Array::try_from_op(|res| unsafe {
            mlx_rust_sys::mlx_astype(
                res,
                self.handle,
                mlx_rust_sys::mlx_dtype_(dtype as u32),
                stream.as_ref().handle,
            )
        })
    }

    /// Borrow the contents without conversion or copying.
    ///
    /// Evaluates first, so this is a synchronization point. Prefer it over
    /// [`to_vec_f32`](Self::to_vec_f32_device) for integer data, which `f32`
    /// cannot represent exactly above 2^24.
    ///
    /// # Errors
    ///
    /// If `Value::DTYPE` is not this array's dtype, or the layout is not
    /// row-contiguous; see [`contiguous`](Self::contiguous_device).
    pub fn as_slice<Value: Element>(&self) -> Result<&[Value]> {
        let actual_dtype = self.dtype()?;
        if actual_dtype != Value::DTYPE {
            return Err(Error::Invalid(format!(
                "array dtype is {actual_dtype:?} but {:?} was requested",
                Value::DTYPE
            )));
        }
        self.eval()?;
        let element_count = self.size();
        if element_count == 0 {
            // MLX returns a null data pointer for a zero-size array.
            return Ok(&[]);
        }
        self.require_row_contiguous()?;
        let data_ptr = unsafe { Value::data_ptr(self.handle) };
        if data_ptr.is_null() {
            return Err(Error::Invalid(
                "array data pointer was null after eval".into(),
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(data_ptr, element_count) })
    }

    /// Copy the contents out without conversion.
    ///
    /// The owning form of [`as_slice`](Self::as_slice), with the same errors.
    pub fn to_vec<Value: Element>(&self) -> Result<Vec<Value>> {
        self.as_slice::<Value>().map(<[Value]>::to_vec)
    }

    /// Copy the contents out as `f32`, converting the dtype if needed.
    ///
    /// Evaluates first, so this is a synchronization point.
    ///
    /// # Errors
    ///
    /// If the layout is not row-contiguous; see
    /// [`contiguous`](Self::contiguous_device).
    #[default_device]
    pub fn to_vec_f32_device(
        &self,
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<f32>> {
        // Owned rather than borrowed: for a non-f32 array the data lives in the
        // temporary `astype` result, which cannot outlive this call.
        let converted = if self.dtype()? == Dtype::Float32 {
            None
        } else {
            Some(self.astype_device(Dtype::Float32, stream)?)
        };
        converted.as_ref().unwrap_or(self).to_vec::<f32>()
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_array_free(self.handle) };
        }
    }
}

impl std::fmt::Debug for Array {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype().ok())
            .finish()
    }
}

/// An owned `mlx_vector_array`, freed on drop.
pub(crate) struct VectorArray {
    handle: mlx_rust_sys::mlx_vector_array,
}

impl VectorArray {
    pub(crate) fn new() -> Self {
        install();
        VectorArray {
            handle: unsafe { mlx_rust_sys::mlx_vector_array_new() },
        }
    }

    pub(crate) fn handle(&self) -> mlx_rust_sys::mlx_vector_array {
        self.handle
    }

    /// For ops that take the vector as an out-parameter.
    pub(crate) fn handle_mut(&mut self) -> &mut mlx_rust_sys::mlx_vector_array {
        &mut self.handle
    }

    pub(crate) fn len(&self) -> usize {
        unsafe { mlx_rust_sys::mlx_vector_array_size(self.handle) }
    }

    pub(crate) fn push(
        &mut self,
        array: &Array,
    ) -> Result<()> {
        check(|| unsafe {
            mlx_rust_sys::mlx_vector_array_append_value(
                self.handle,
                array.handle,
            )
        })
    }

    pub(crate) fn get(
        &self,
        index: usize,
    ) -> Result<Array> {
        let mut out = Array::empty();
        check(|| unsafe {
            mlx_rust_sys::mlx_vector_array_get(
                &mut out.handle,
                self.handle,
                index,
            )
        })?;
        Ok(out)
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_vector_array_free(self.handle) };
        }
    }
}

/// Evaluate several arrays in one call, paying one synchronization for the batch.
pub fn eval_all(arrays: &[&Array]) -> Result<()> {
    let mut vector = VectorArray::new();
    for array in arrays {
        vector.push(array)?;
    }
    check(|| unsafe { mlx_rust_sys::mlx_eval(vector.handle()) })
}
