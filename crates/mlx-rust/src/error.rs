use std::{
    cell::RefCell,
    ffi::{CStr, c_char, c_void},
    sync::Once,
};

/// An error from MLX or from this crate's preconditions.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An mlx-c call returned non-zero.
    #[error("mlx: {message}")]
    Mlx {
        /// What MLX passed to the error handler, or a fallback naming the status.
        message: String,
    },

    /// A precondition failed before reaching MLX.
    #[error("{0}")]
    Invalid(String),

    /// An MLX array had a dtype this crate does not map.
    #[error("mlx: unsupported dtype code {0}")]
    UnsupportedDtype(u32),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn error_handler(
    message: *const c_char,
    _handler_data: *mut c_void,
) {
    let message = if message.is_null() {
        String::from("(null)")
    } else {
        unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
    };
    LAST_ERROR.with(|last_error| *last_error.borrow_mut() = Some(message));
}

static INSTALL: Once = Once::new();

/// Replace mlx-c's default error handler, which prints and calls `exit(-1)`.
///
/// Idempotent, and called for you by everything in this crate. Call it directly
/// only before invoking `mlx-rust-sys` yourself, for an op this crate does not wrap.
pub fn install() {
    INSTALL.call_once(|| unsafe {
        mlx_rust_sys::mlx_set_error_handler(
            Some(error_handler),
            std::ptr::null_mut(),
            None,
        );
    });
}

pub(crate) fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|last_error| last_error.borrow_mut().take())
}

pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|last_error| *last_error.borrow_mut() = None);
}

/// Run an mlx-c call that returns a status code, converting it to a `Result`.
///
/// Clears the error slot first so a stale message is not attributed to this call.
pub(crate) fn check(call: impl FnOnce() -> i32) -> Result<()> {
    install();
    clear_last_error();
    let status = call();
    if status == 0 {
        return Ok(());
    }
    Err(Error::Mlx {
        message: take_last_error()
            .unwrap_or_else(|| format!("call failed with status {status}")),
    })
}
