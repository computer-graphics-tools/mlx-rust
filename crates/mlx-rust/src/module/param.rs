//! Marking struct fields as parameters.
//!
//! Adapted from [mlx-rs](https://github.com/oxiglade/mlx-rs) (MIT OR
//! Apache-2.0); see NOTICE.

use std::ops::{Deref, DerefMut};

use super::{ModuleParameters, nested::NestedValue};
use crate::Array;

/// A trainable parameter, or a group of them.
///
/// Implemented for [`Array`], `Option<Array>`, `Vec<T>` and any
/// [`ModuleParameters`], so a field of any of those shapes can be marked
/// `#[param]`.
pub trait Parameter {
    /// How many arrays this holds.
    fn count(&self) -> usize;

    /// Stop gradients flowing to these arrays.
    fn freeze(
        &mut self,
        recursive: bool,
    );

    /// Let gradients flow to these arrays again.
    fn unfreeze(
        &mut self,
        recursive: bool,
    );

    /// `None` when this holds no arrays at all, so a caller can tell "all frozen"
    /// apart from "nothing to freeze".
    fn is_frozen(&self) -> Option<bool>;

    /// Borrow as a tree.
    fn as_nested_value(&self) -> NestedValue<&str, &Array>;

    /// Borrow mutably as a tree.
    fn as_nested_value_mut(&mut self) -> NestedValue<&str, &mut Array>;

    /// Borrow as a tree, skipping the frozen entries.
    fn as_trainable_nested_value(&self) -> Option<NestedValue<&str, &Array>>;
}

/// A struct field holding parameters.
///
/// Derefs to its contents, so a `Param<Array>` is used as an `Array`.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The parameters themselves.
    pub value: T,
    frozen: bool,
}

impl<T> Param<T> {
    /// Wrap `value` as a trainable parameter.
    pub fn new(value: T) -> Self {
        Param {
            value,
            frozen: false,
        }
    }
}

impl<T> From<T> for Param<T> {
    fn from(value: T) -> Self {
        Param::new(value)
    }
}

impl<T> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for Param<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> AsRef<T> for Param<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl Parameter for Param<Array> {
    fn count(&self) -> usize {
        1
    }

    fn freeze(
        &mut self,
        _recursive: bool,
    ) {
        self.frozen = true;
    }

    fn unfreeze(
        &mut self,
        _recursive: bool,
    ) {
        self.frozen = false;
    }

    fn is_frozen(&self) -> Option<bool> {
        Some(self.frozen)
    }

    fn as_nested_value(&self) -> NestedValue<&str, &Array> {
        NestedValue::Value(&self.value)
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<&str, &mut Array> {
        NestedValue::Value(&mut self.value)
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<&str, &Array>> {
        (!self.frozen).then(|| NestedValue::Value(&self.value))
    }
}

impl Parameter for Param<Option<Array>> {
    fn count(&self) -> usize {
        usize::from(self.value.is_some())
    }

    fn freeze(
        &mut self,
        _recursive: bool,
    ) {
        self.frozen = true;
    }

    fn unfreeze(
        &mut self,
        _recursive: bool,
    ) {
        self.frozen = false;
    }

    fn is_frozen(&self) -> Option<bool> {
        self.value.as_ref().map(|_| self.frozen)
    }

    fn as_nested_value(&self) -> NestedValue<&str, &Array> {
        match &self.value {
            Some(array) => NestedValue::Value(array),
            None => NestedValue::Map(indexmap::IndexMap::new()),
        }
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<&str, &mut Array> {
        match &mut self.value {
            Some(array) => NestedValue::Value(array),
            None => NestedValue::Map(indexmap::IndexMap::new()),
        }
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<&str, &Array>> {
        match (&self.value, self.frozen) {
            (Some(array), false) => Some(NestedValue::Value(array)),
            _ => None,
        }
    }
}

/// A nested module is itself a parameter group.
impl<T: ModuleParameters> Parameter for T {
    fn count(&self) -> usize {
        self.num_parameters()
    }

    fn freeze(
        &mut self,
        recursive: bool,
    ) {
        self.freeze_parameters(recursive);
    }

    fn unfreeze(
        &mut self,
        recursive: bool,
    ) {
        self.unfreeze_parameters(recursive);
    }

    fn is_frozen(&self) -> Option<bool> {
        self.all_frozen()
    }

    fn as_nested_value(&self) -> NestedValue<&str, &Array> {
        self.parameters().into()
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<&str, &mut Array> {
        self.parameters_mut().into()
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<&str, &Array>> {
        Some(self.trainable_parameters().into())
    }
}
