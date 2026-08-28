use crate::error::{Result, check, install};

/// An MLX execution stream.
///
/// Neither `Send` nor `Sync`: MLX's default streams are process-global.
pub struct Stream {
    pub(crate) handle: mlx_rust_sys::mlx_stream,
}

thread_local! {
    /// Cached so the defaulted ops do not allocate a handle per call. Every
    /// handle refers to the same underlying stream, so this is not observable.
    static DEFAULT_STREAM: Stream = Stream::gpu();
    static DEFAULT_CPU_STREAM: Stream = Stream::cpu();
}

impl Stream {
    /// Run `action` with the default stream, [`Stream::gpu`]. This is what the
    /// ops without an explicit stream argument use.
    pub fn with_default<Output>(
        action: impl FnOnce(&Stream) -> Output
    ) -> Output {
        DEFAULT_STREAM.with(action)
    }

    /// Run `action` with the default CPU stream.
    ///
    /// Used by the ops MLX implements only on the CPU, such as most of
    /// [`linalg`](crate::linalg): defaulting those to the GPU would make every
    /// call fail.
    pub fn with_default_cpu<Output>(
        action: impl FnOnce(&Stream) -> Output
    ) -> Output {
        DEFAULT_CPU_STREAM.with(action)
    }

    /// The default GPU stream.
    pub fn gpu() -> Self {
        install();
        Stream {
            handle: unsafe { mlx_rust_sys::mlx_default_gpu_stream_new() },
        }
    }

    /// The default CPU stream.
    pub fn cpu() -> Self {
        install();
        Stream {
            handle: unsafe { mlx_rust_sys::mlx_default_cpu_stream_new() },
        }
    }

    /// Block until every operation queued on this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        check(|| unsafe { mlx_rust_sys::mlx_synchronize(self.handle) })
    }
}

impl AsRef<Stream> for Stream {
    fn as_ref(&self) -> &Stream {
        self
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.handle.ctx.is_null() {
            unsafe { mlx_rust_sys::mlx_stream_free(self.handle) };
        }
    }
}
