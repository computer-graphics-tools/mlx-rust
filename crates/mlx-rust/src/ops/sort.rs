//! Sorting.

use super::macros::unary_ops;

unary_ops! {
    /// Sort along the last axis.
    sort_device => mlx_sort,
    /// Indices that would sort along the last axis.
    argsort_device => mlx_argsort,
}
