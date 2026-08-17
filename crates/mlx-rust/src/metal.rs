use std::ffi::CString;

use crate::error::{Error, Result, check, install};

/// Whether this build has a working Metal backend and a reachable device.
///
/// `false` when the `metal` feature is off, or when no device is available --
/// as in a headless session.
pub fn is_available() -> bool {
    install();
    let mut available = false;
    let queried =
        unsafe { mlx_rust_sys::mlx_metal_is_available(&mut available) } == 0;
    queried && available
}

/// Begin a Metal capture, writing a `.gputrace` to `path`.
///
/// # Errors
///
/// If capture is unavailable, as in a headless session.
pub fn start_capture(path: &str) -> Result<()> {
    let capture_path = CString::new(path)
        .map_err(|_| Error::Invalid("capture path contains a NUL".into()))?;
    check(|| unsafe {
        mlx_rust_sys::mlx_metal_start_capture(capture_path.as_ptr())
    })
}

/// End the capture started by [`start_capture`] and flush the trace to disk.
pub fn stop_capture() -> Result<()> {
    check(|| unsafe { mlx_rust_sys::mlx_metal_stop_capture() })
}

/// Path of the `mlx.metallib` this crate was built against.
#[cfg(feature = "metal")]
pub fn metallib_path() -> &'static str {
    mlx_rust_sys::METALLIB_PATH
}

/// MLX allocator statistics, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Held by live arrays.
    pub active: usize,
    /// Held in the free-buffer cache.
    pub cached: usize,
    /// High-water mark of `active`.
    pub peak: usize,
    /// The allocator's ceiling.
    pub limit: usize,
}

/// Read MLX's allocator counters.
pub fn memory_stats() -> Result<MemoryStats> {
    let (mut active, mut cached, mut peak, mut limit) = (0, 0, 0, 0);
    check(|| unsafe { mlx_rust_sys::mlx_get_active_memory(&mut active) })?;
    check(|| unsafe { mlx_rust_sys::mlx_get_cache_memory(&mut cached) })?;
    check(|| unsafe { mlx_rust_sys::mlx_get_peak_memory(&mut peak) })?;
    check(|| unsafe { mlx_rust_sys::mlx_get_memory_limit(&mut limit) })?;
    Ok(MemoryStats {
        active,
        cached,
        peak,
        limit,
    })
}

/// Return cached-but-unused buffers to the system.
///
/// MLX caches freed buffers rather than releasing them, which hides the true
/// footprint of the next allocation.
pub fn clear_cache() -> Result<()> {
    check(|| unsafe { mlx_rust_sys::mlx_clear_cache() })
}

/// Set the byte limit on MLX's allocator. Returns the previous value.
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    let mut previous_limit = 0usize;
    check(|| unsafe {
        mlx_rust_sys::mlx_set_memory_limit(&mut previous_limit, limit)
    })?;
    Ok(previous_limit)
}
