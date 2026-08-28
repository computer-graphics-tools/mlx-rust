use half::{bf16, f16};
use num_complex::Complex32;

use crate::error::{Error, Result};

/// Declares [`Dtype`] and `from_raw` from one table, so they cannot drift.
macro_rules! dtypes {
    ($($variant:ident = $raw:literal,)+) => {
        /// Element type of an [`Array`](crate::Array).
        ///
        /// Discriminants match `mlx_dtype`, so `as u32` round-trips.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        #[repr(u32)]
        #[expect(missing_docs, reason = "variant names are the documentation")]
        pub enum Dtype {
            $($variant = $raw,)+
        }

        impl Dtype {
            /// Convert an `mlx_dtype` discriminant.
            ///
            /// # Errors
            ///
            /// If `raw` is not a dtype this crate maps.
            pub fn from_raw(raw: u32) -> Result<Self> {
                Ok(match raw {
                    $($raw => Dtype::$variant,)+
                    other => return Err(Error::UnsupportedDtype(other)),
                })
            }
        }
    };
}

dtypes! {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Bfloat16 = 12,
    Complex64 = 13,
}

/// A Rust type that can be moved between host memory and an MLX array.
///
/// # Safety
///
/// `DTYPE` must describe exactly the in-memory representation of `Self`, `Self`
/// must be plain-old-data with no padding, and `data_ptr` must return the buffer
/// belonging to an array of `DTYPE`.
pub unsafe trait Element: Copy + 'static {
    /// The MLX dtype whose in-memory representation is exactly `Self`.
    const DTYPE: Dtype;

    /// Pointer to the array's element buffer.
    ///
    /// # Safety
    ///
    /// `handle` must be a live, evaluated array whose dtype is `Self::DTYPE`.
    unsafe fn data_ptr(handle: mlx_rust_sys::mlx_array) -> *const Self;
}

/// Implements [`Element`] for `$rust_type` via mlx-c's `$accessor`.
///
/// `data_ptr` is a raw-pointer cast, which compiles for any pointee, so
/// `$ffi_type` names the accessor's pointee and the width is asserted here
/// instead.
macro_rules! element {
    ($rust_type:ty, $dtype:expr, $accessor:ident, $ffi_type:ty) => {
        const _: () = assert!(
            size_of::<$rust_type>() == size_of::<$ffi_type>(),
            concat!(
                "mlx-c's ",
                stringify!($accessor),
                " element width no longer matches ",
                stringify!($rust_type),
            )
        );

        unsafe impl Element for $rust_type {
            const DTYPE: Dtype = $dtype;

            unsafe fn data_ptr(handle: mlx_rust_sys::mlx_array) -> *const Self {
                unsafe { mlx_rust_sys::$accessor(handle).cast::<Self>() }
            }
        }
    };
}

element!(bool, Dtype::Bool, mlx_array_data_bool, bool);
element!(u8, Dtype::Uint8, mlx_array_data_uint8, u8);
element!(u16, Dtype::Uint16, mlx_array_data_uint16, u16);
element!(u32, Dtype::Uint32, mlx_array_data_uint32, u32);
element!(u64, Dtype::Uint64, mlx_array_data_uint64, u64);
element!(i8, Dtype::Int8, mlx_array_data_int8, i8);
element!(i16, Dtype::Int16, mlx_array_data_int16, i16);
element!(i32, Dtype::Int32, mlx_array_data_int32, i32);
element!(i64, Dtype::Int64, mlx_array_data_int64, i64);
element!(f32, Dtype::Float32, mlx_array_data_float32, f32);
element!(f64, Dtype::Float64, mlx_array_data_float64, f64);
element!(f16, Dtype::Float16, mlx_array_data_float16, mlx_rust_sys::float16_t);
element!(
    bf16,
    Dtype::Bfloat16,
    mlx_array_data_bfloat16,
    mlx_rust_sys::bfloat16_t
);
// `Complex32` and mlx-c's `__BindgenComplex<f32>` are both `#[repr(C)] {re, im}`;
// the width assertion above proves it.
element!(
    Complex32,
    Dtype::Complex64,
    mlx_array_data_complex64,
    mlx_rust_sys::mlx_complex64_t
);
