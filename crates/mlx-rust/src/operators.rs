//! `std::ops` impls, so `&a + &b` reads as it does in MLX's Python API.
//!
//! Implemented on `&Array` rather than `Array`: MLX arrays are refcounted
//! handles, and taking them by value would move an operand a caller almost always
//! wants to keep. These panic on a shape mismatch; use [`ops::add`](crate::ops)
//! and friends for the fallible form.

use std::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};

use crate::{Array, ops};

macro_rules! binary_operator {
    ($($trait:ident, $method:ident => $op:path, $symbol:literal;)*) => {$(
        impl $trait<&Array> for &Array {
            type Output = Array;

            /// # Panics
            ///
            /// If the shapes do not broadcast.
            fn $method(
                self,
                rhs: &Array,
            ) -> Array {
                $op(self, rhs).unwrap_or_else(|error| {
                    panic!("{} {} {}: {error}", "lhs", $symbol, "rhs")
                })
            }
        }
    )*};
}

binary_operator! {
    Add, add => ops::add, "+";
    Sub, sub => ops::subtract, "-";
    Mul, mul => ops::multiply, "*";
    Div, div => ops::divide, "/";
    Rem, rem => ops::remainder, "%";
}

impl Neg for &Array {
    type Output = Array;

    /// # Panics
    ///
    /// If MLX rejects the dtype.
    fn neg(self) -> Array {
        ops::negative(self).unwrap_or_else(|error| panic!("-lhs: {error}"))
    }
}

impl Not for &Array {
    type Output = Array;

    /// # Panics
    ///
    /// If MLX rejects the dtype.
    fn not(self) -> Array {
        ops::logical_not(self).unwrap_or_else(|error| panic!("!lhs: {error}"))
    }
}
