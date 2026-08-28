//! Saving and loading arrays, mirroring `mlx.core`'s `save` and `load`.

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    path::Path,
};

use mlx_rust_macros::default_device;

use crate::{
    array::Array,
    error::{Error, Result, check, install},
    stream::Stream,
};

fn path_to_cstring(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error::Invalid("path contains a NUL".into()))
}

/// An owned `mlx_map_string_to_array`.
struct ArrayMap {
    handle: mlx_rust_sys::mlx_map_string_to_array,
}

impl ArrayMap {
    fn new() -> Self {
        install();
        ArrayMap {
            handle: unsafe { mlx_rust_sys::mlx_map_string_to_array_new() },
        }
    }

    fn insert(
        &mut self,
        key: &str,
        value: &Array,
    ) -> Result<()> {
        let key = CString::new(key)
            .map_err(|_| Error::Invalid("key contains a NUL".into()))?;
        check(|| unsafe {
            mlx_rust_sys::mlx_map_string_to_array_insert(
                self.handle,
                key.as_ptr(),
                value.handle,
            )
        })
    }

    /// Drain into a Rust map. Iteration is by mlx-c's iterator, which reports
    /// exhaustion with a positive status rather than an error.
    fn into_hash_map(self) -> Result<HashMap<String, Array>> {
        let mut entries = HashMap::new();
        let iterator = unsafe {
            mlx_rust_sys::mlx_map_string_to_array_iterator_new(self.handle)
        };
        loop {
            let mut key: *const std::ffi::c_char = std::ptr::null();
            let mut value = Array::empty();
            let status = unsafe {
                mlx_rust_sys::mlx_map_string_to_array_iterator_next(
                    &mut key,
                    &mut value.handle,
                    iterator,
                )
            };
            if status != 0 {
                break;
            }
            let name =
                unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned();
            entries.insert(name, value);
        }
        unsafe {
            mlx_rust_sys::mlx_map_string_to_array_iterator_free(iterator);
        }
        Ok(entries)
    }
}

impl Drop for ArrayMap {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_map_string_to_array_free(self.handle) };
        }
    }
}

/// An owned `mlx_map_string_to_string`, for safetensors metadata.
struct StringMap {
    handle: mlx_rust_sys::mlx_map_string_to_string,
}

impl StringMap {
    fn new() -> Self {
        install();
        StringMap {
            handle: unsafe { mlx_rust_sys::mlx_map_string_to_string_new() },
        }
    }

    fn insert(
        &mut self,
        key: &str,
        value: &str,
    ) -> Result<()> {
        let key = CString::new(key)
            .map_err(|_| Error::Invalid("key contains a NUL".into()))?;
        let value = CString::new(value)
            .map_err(|_| Error::Invalid("value contains a NUL".into()))?;
        check(|| unsafe {
            mlx_rust_sys::mlx_map_string_to_string_insert(
                self.handle,
                key.as_ptr(),
                value.as_ptr(),
            )
        })
    }

    fn into_hash_map(self) -> HashMap<String, String> {
        let mut entries = HashMap::new();
        let iterator = unsafe {
            mlx_rust_sys::mlx_map_string_to_string_iterator_new(self.handle)
        };
        loop {
            let mut key: *const std::ffi::c_char = std::ptr::null();
            let mut value: *const std::ffi::c_char = std::ptr::null();
            let status = unsafe {
                mlx_rust_sys::mlx_map_string_to_string_iterator_next(
                    &mut key, &mut value, iterator,
                )
            };
            if status != 0 {
                break;
            }
            entries.insert(
                unsafe { CStr::from_ptr(key) }.to_string_lossy().into_owned(),
                unsafe { CStr::from_ptr(value) }.to_string_lossy().into_owned(),
            );
        }
        unsafe {
            mlx_rust_sys::mlx_map_string_to_string_iterator_free(iterator);
        }
        entries
    }
}

impl Drop for StringMap {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_map_string_to_string_free(self.handle) };
        }
    }
}

/// Write one array to a `.npy` file.
pub fn save(
    file: &Path,
    array: impl AsRef<Array>,
) -> Result<()> {
    let file = path_to_cstring(file)?;
    check(|| unsafe {
        mlx_rust_sys::mlx_save(file.as_ptr(), array.as_ref().handle)
    })
}

/// Read one array from a `.npy` file.
#[default_device(cpu)]
pub fn load_device(
    file: &Path,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let file = path_to_cstring(file)?;
    Array::try_from_op(|result| unsafe {
        mlx_rust_sys::mlx_load(result, file.as_ptr(), stream.as_ref().handle)
    })
}

/// Write named arrays, and optional metadata, to a `.safetensors` file.
pub fn save_safetensors(
    file: &Path,
    arrays: &HashMap<String, Array>,
    metadata: &HashMap<String, String>,
) -> Result<()> {
    let file = path_to_cstring(file)?;
    let mut parameters = ArrayMap::new();
    for (name, array) in arrays {
        parameters.insert(name, array)?;
    }
    let mut extra = StringMap::new();
    for (key, value) in metadata {
        extra.insert(key, value)?;
    }
    check(|| unsafe {
        mlx_rust_sys::mlx_save_safetensors(
            file.as_ptr(),
            parameters.handle,
            extra.handle,
        )
    })
}

/// Read named arrays and metadata from a `.safetensors` file.
#[default_device(cpu)]
pub fn load_safetensors_device(
    file: &Path,
    stream: impl AsRef<Stream>,
) -> Result<(HashMap<String, Array>, HashMap<String, String>)> {
    let file = path_to_cstring(file)?;
    let mut arrays = ArrayMap::new();
    let mut metadata = StringMap::new();
    check(|| unsafe {
        mlx_rust_sys::mlx_load_safetensors(
            &mut arrays.handle,
            &mut metadata.handle,
            file.as_ptr(),
            stream.as_ref().handle,
        )
    })?;
    Ok((arrays.into_hash_map()?, metadata.into_hash_map()))
}
