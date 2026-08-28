//! Modules and their parameters.
//!
//! Adapted from [mlx-rs](https://github.com/oxiglade/mlx-rs) (MIT OR
//! Apache-2.0); see NOTICE.
//!
//! Rust cannot enumerate a struct's fields at runtime the way Python reads
//! `__dict__`, so parameters are declared: mark each field `#[param]` and derive
//! `ModuleParameters`. The derive walks the marked fields and nothing else.

pub mod nested;
pub mod param;

use std::{collections::HashMap, path::Path};

use indexmap::IndexMap;
pub use mlx_rust_macros::ModuleParameters;
pub use nested::{NestedMap, NestedValue};
pub use param::{Param, Parameter};

use crate::{Array, error::Result, io};

/// A borrowed parameter tree.
pub type ModuleParamRef<'a> = NestedMap<&'a str, &'a Array>;
/// A mutably borrowed parameter tree.
pub type ModuleParamMut<'a> = NestedMap<&'a str, &'a mut Array>;
/// A parameter tree flattened to `.`-joined keys, in insertion order.
pub type FlatParams<V> = IndexMap<String, V>;

/// A type whose parameters can be enumerated, frozen and replaced.
///
/// Derive this rather than implementing it: `#[derive(ModuleParameters)]` with
/// `#[param]` on each parameter-holding field.
pub trait ModuleParameters {
    /// Borrow every parameter.
    fn parameters(&self) -> ModuleParamRef<'_>;

    /// Borrow every parameter mutably.
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;

    /// Borrow only the parameters gradients still flow to.
    fn trainable_parameters(&self) -> ModuleParamRef<'_>;

    /// Stop gradients flowing to every parameter.
    fn freeze_parameters(
        &mut self,
        recursive: bool,
    );

    /// Let gradients flow to every parameter again.
    fn unfreeze_parameters(
        &mut self,
        recursive: bool,
    );

    /// `Some(true)` when every parameter is frozen, `None` when there are none.
    fn all_frozen(&self) -> Option<bool>;

    /// `Some(true)` when any parameter is frozen, `None` when there are none.
    fn any_frozen(&self) -> Option<bool>;

    /// How many arrays this module and its children hold.
    fn num_parameters(&self) -> usize;

    /// Every parameter under a `.`-joined key, in declaration order.
    fn flatten(&self) -> FlatParams<&Array> {
        self.parameters().flatten()
    }

    /// Replace parameters by flattened key. Keys with no matching parameter are
    /// ignored, which is what lets a checkpoint carry extra entries.
    fn update_flattened(
        &mut self,
        replacements: &FlatParams<Array>,
    ) {
        for (key, slot) in self.parameters_mut().flatten() {
            if let Some(replacement) = replacements.get(&key) {
                *slot = replacement.clone_handle();
            }
        }
    }

    /// Write every parameter to a `.safetensors` file.
    fn save_safetensors(
        &self,
        file: &Path,
    ) -> Result<()> {
        let owned: HashMap<String, Array> = self
            .flatten()
            .into_iter()
            .map(|(key, array)| (key, array.clone_handle()))
            .collect();
        io::save_safetensors(file, &owned, &HashMap::new())
    }

    /// Load every parameter from a `.safetensors` file.
    fn load_safetensors(
        &mut self,
        file: &Path,
    ) -> Result<()> {
        let (loaded, _metadata) = io::load_safetensors(file)?;
        let ordered: FlatParams<Array> = loaded.into_iter().collect();
        self.update_flattened(&ordered);
        Ok(())
    }
}

/// A module that maps an input to an output.
///
/// `forward` takes `&mut self` because stateful layers -- Dropout's RNG,
/// BatchNorm's running statistics -- mutate as they run.
pub trait Module<Input>: ModuleParameters + std::fmt::Debug {
    /// What `forward` produces.
    type Output;

    /// How `forward` fails.
    type Error: std::error::Error;

    /// Run the module.
    fn forward(
        &mut self,
        input: Input,
    ) -> std::result::Result<Self::Output, Self::Error>;

    /// Switch between training and evaluation behaviour, recursively.
    fn training_mode(
        &mut self,
        mode: bool,
    );
}

/// The common case: `&Array` in, `Array` out.
pub trait UnaryModule:
    for<'a> Module<&'a Array, Output = Array, Error = crate::Error>
{
}

impl<T> UnaryModule for T where
    T: for<'a> Module<&'a Array, Output = Array, Error = crate::Error>
{
}
