//! Operations on [`Array`](crate::Array), mirroring `mlx.core`.
//!
//! Every op has a `_device` twin taking an explicit [`Stream`](crate::Stream);
//! the plain form uses the default stream.

mod macros;
pub(crate) mod optional;
pub(crate) use optional::optional_float;

pub mod arithmetic;
pub mod convolution;
pub mod cumulative;
pub mod factory;
pub mod indexing;
pub mod linear;
pub mod logical;
pub mod manipulation;
pub mod misc;
pub mod ordering;
pub mod quantization;
pub mod reduction;
pub mod shapes;
pub mod slicing;
pub mod sort;

pub use arithmetic::*;
pub use convolution::*;
pub use cumulative::*;
pub use factory::*;
pub use indexing::*;
pub use linear::*;
pub use logical::*;
pub use manipulation::*;
pub use misc::*;
pub use ordering::*;
pub use quantization::*;
pub use reduction::*;
pub use shapes::*;
pub use slicing::*;
pub use sort::*;
