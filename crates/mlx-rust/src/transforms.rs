//! Function transformations, mirroring `mlx.core`'s `value_and_grad`, `grad`,
//! `vjp` and `jvp`.
//!
//! MLX takes its functions as `mlx_closure`, a C function pointer plus an opaque
//! payload. A Rust closure is boxed into that payload and reached through a
//! trampoline, which catches unwinds so a panic cannot cross the FFI boundary.

use std::{ffi::c_void, panic::AssertUnwindSafe};

use crate::{
    array::{Array, VectorArray},
    error::{Result, check, install},
};

/// The boxed Rust closure a [`Closure`] carries as its mlx-c payload.
type BoxedFn = Box<dyn FnMut(&[Array]) -> Result<Vec<Array>>>;

/// An `mlx_closure` owning a boxed Rust function.
pub struct Closure {
    handle: mlx_rust_sys::mlx_closure,
}

impl Closure {
    /// Call this closure on `inputs`.
    pub fn apply(
        &self,
        inputs: &[&Array],
    ) -> Result<Vec<Array>> {
        let inputs = to_vector(inputs)?;
        let mut outputs = VectorArray::new();
        check(|| unsafe {
            mlx_rust_sys::mlx_closure_apply(
                outputs.handle_mut(),
                self.handle,
                inputs.handle(),
            )
        })?;
        from_vector(&outputs)
    }

    /// Wrap a Rust function so MLX can call it.
    pub fn new(
        function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static
    ) -> Self {
        install();
        let payload: *mut BoxedFn = Box::into_raw(Box::new(Box::new(function)));
        Closure {
            handle: unsafe {
                mlx_rust_sys::mlx_closure_new_func_payload(
                    Some(trampoline),
                    payload.cast::<c_void>(),
                    Some(drop_payload),
                )
            },
        }
    }
}

impl Drop for Closure {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_closure_free(self.handle) };
        }
    }
}

/// Free the boxed closure once MLX is done with it.
unsafe extern "C" fn drop_payload(payload: *mut c_void) {
    if !payload.is_null() {
        drop(unsafe { Box::from_raw(payload.cast::<BoxedFn>()) });
    }
}

/// Call the boxed closure on MLX's behalf.
///
/// Returns non-zero rather than unwinding: a panic crossing back into C++ would
/// be undefined behaviour.
unsafe extern "C" fn trampoline(
    result: *mut mlx_rust_sys::mlx_vector_array,
    input: mlx_rust_sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let function = unsafe { &mut *payload.cast::<BoxedFn>() };
        let inputs = borrowed_arrays(input)?;
        let outputs = function(&inputs)?;

        let mut vector = VectorArray::new();
        for array in &outputs {
            vector.push(array)?;
        }
        check(|| unsafe {
            mlx_rust_sys::mlx_vector_array_set(result, vector.handle())
        })
    }));
    match outcome {
        Ok(Ok(())) => 0,
        _ => 1,
    }
}

/// Copy the arrays out of a vector MLX owns. The copies are refcounted, so this
/// does not take ownership of the caller's vector.
fn borrowed_arrays(
    vector: mlx_rust_sys::mlx_vector_array
) -> Result<Vec<Array>> {
    let count = unsafe { mlx_rust_sys::mlx_vector_array_size(vector) };
    (0..count)
        .map(|index| {
            let mut array = Array::empty();
            check(|| unsafe {
                mlx_rust_sys::mlx_vector_array_get(
                    &mut array.handle,
                    vector,
                    index,
                )
            })?;
            Ok(array)
        })
        .collect()
}

fn to_vector(arrays: &[&Array]) -> Result<VectorArray> {
    let mut vector = VectorArray::new();
    for array in arrays {
        vector.push(array)?;
    }
    Ok(vector)
}

fn from_vector(vector: &VectorArray) -> Result<Vec<Array>> {
    (0..vector.len()).map(|index| vector.get(index)).collect()
}

/// A function returning both its value and its gradient.
pub struct ValueAndGrad {
    handle: mlx_rust_sys::mlx_closure_value_and_grad,
    // The closure must outlive the transform that calls it.
    _function: Closure,
}

impl ValueAndGrad {
    /// Evaluate at `primals`, returning `(values, gradients)`.
    pub fn apply(
        &self,
        primals: &[&Array],
    ) -> Result<(Vec<Array>, Vec<Array>)> {
        let inputs = to_vector(primals)?;
        let mut values = VectorArray::new();
        let mut gradients = VectorArray::new();
        check(|| unsafe {
            mlx_rust_sys::mlx_closure_value_and_grad_apply(
                values.handle_mut(),
                gradients.handle_mut(),
                self.handle,
                inputs.handle(),
            )
        })?;
        Ok((from_vector(&values)?, from_vector(&gradients)?))
    }
}

impl Drop for ValueAndGrad {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe {
                mlx_rust_sys::mlx_closure_value_and_grad_free(self.handle)
            };
        }
    }
}

/// Transform `function` into one returning its value and its gradient with
/// respect to the arguments at `argnums`.
pub fn value_and_grad(
    function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    argnums: &[i32],
) -> Result<ValueAndGrad> {
    let function = Closure::new(function);
    let mut handle = unsafe { mlx_rust_sys::mlx_closure_value_and_grad_new() };
    check(|| unsafe {
        mlx_rust_sys::mlx_value_and_grad(
            &mut handle,
            function.handle,
            argnums.as_ptr(),
            argnums.len(),
        )
    })?;
    Ok(ValueAndGrad {
        handle,
        _function: function,
    })
}

/// The gradient of `function` with respect to the arguments at `argnums`.
///
/// `function` must return a single **scalar** array -- shape `[]`, not `[1]`.
/// Reduce with [`sum`](crate::ops::sum) or similar first if it does not.
pub fn grad(
    function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    argnums: &[i32],
) -> Result<impl Fn(&[&Array]) -> Result<Vec<Array>>> {
    let transformed = value_and_grad(function, argnums)?;
    Ok(move |primals: &[&Array]| {
        transformed.apply(primals).map(|(_, gradients)| gradients)
    })
}

/// Vector-Jacobian product: `(outputs, cotangent_gradients)`.
pub fn vjp(
    function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    primals: &[&Array],
    cotangents: &[&Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let function = Closure::new(function);
    let primals = to_vector(primals)?;
    let cotangents = to_vector(cotangents)?;
    let mut outputs = VectorArray::new();
    let mut gradients = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_vjp(
            outputs.handle_mut(),
            gradients.handle_mut(),
            function.handle,
            primals.handle(),
            cotangents.handle(),
        )
    })?;
    Ok((from_vector(&outputs)?, from_vector(&gradients)?))
}

/// Jacobian-vector product: `(outputs, tangent_outputs)`.
pub fn jvp(
    function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    primals: &[&Array],
    tangents: &[&Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let function = Closure::new(function);
    let primals = to_vector(primals)?;
    let tangents = to_vector(tangents)?;
    let mut outputs = VectorArray::new();
    let mut tangent_outputs = VectorArray::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_jvp(
            outputs.handle_mut(),
            tangent_outputs.handle_mut(),
            function.handle,
            primals.handle(),
            tangents.handle(),
        )
    })?;
    Ok((from_vector(&outputs)?, from_vector(&tangent_outputs)?))
}

/// Trace and cache `function`, so repeat calls reuse one compiled graph.
///
/// With `shapeless`, the trace is reused across input shapes; otherwise MLX
/// retraces when a shape changes.
pub fn compile(
    function: impl FnMut(&[Array]) -> Result<Vec<Array>> + 'static,
    shapeless: bool,
) -> Result<Closure> {
    let function = Closure::new(function);
    let mut handle = unsafe { mlx_rust_sys::mlx_closure_new() };
    check(|| unsafe {
        mlx_rust_sys::mlx_compile(&mut handle, function.handle, shapeless)
    })?;
    Ok(Closure {
        handle,
    })
}

/// Turn graph compilation on globally.
pub fn enable_compile() -> Result<()> {
    install();
    check(|| unsafe { mlx_rust_sys::mlx_enable_compile() })
}

/// Turn graph compilation off globally.
pub fn disable_compile() -> Result<()> {
    install();
    check(|| unsafe { mlx_rust_sys::mlx_disable_compile() })
}
