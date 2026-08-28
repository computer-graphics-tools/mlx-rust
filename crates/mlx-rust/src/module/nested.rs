//! A nested parameter tree.
//!
//! Adapted from [mlx-rs](https://github.com/oxiglade/mlx-rs) (MIT OR
//! Apache-2.0); see NOTICE. Keyed by [`IndexMap`] rather than `HashMap` so that
//! flattening is insertion-ordered and checkpoint output is reproducible.

use std::fmt::Display;

use indexmap::IndexMap;

/// Separates the levels of a flattened key, as MLX's Python API does.
const DELIMITER: char = '.';

/// Either a leaf value or a sub-tree.
#[derive(Debug, Clone)]
pub enum NestedValue<K, V> {
    /// A leaf.
    Value(V),
    /// A sub-tree.
    Map(IndexMap<K, NestedValue<K, V>>),
}

impl<K, V> NestedValue<K, V>
where
    K: Display,
{
    /// Flatten into `prefix`-joined keys, in insertion order.
    pub fn flatten(
        self,
        prefix: &str,
    ) -> IndexMap<String, V> {
        let mut flattened = IndexMap::new();
        self.flatten_into(prefix, &mut flattened);
        flattened
    }

    fn flatten_into(
        self,
        prefix: &str,
        out: &mut IndexMap<String, V>,
    ) {
        match self {
            NestedValue::Value(value) => {
                out.insert(prefix.to_owned(), value);
            },
            NestedValue::Map(entries) => {
                for (key, value) in entries {
                    let joined = if prefix.is_empty() {
                        key.to_string()
                    } else {
                        format!("{prefix}{DELIMITER}{key}")
                    };
                    value.flatten_into(&joined, out);
                }
            },
        }
    }
}

/// A tree of parameters, one level deep at the root.
#[derive(Debug, Clone)]
pub struct NestedMap<K, V> {
    /// The root entries.
    pub entries: IndexMap<K, NestedValue<K, V>>,
}

impl<K, V> Default for NestedMap<K, V> {
    fn default() -> Self {
        NestedMap::new()
    }
}

impl<K, V> NestedMap<K, V> {
    /// An empty tree.
    pub fn new() -> Self {
        NestedMap {
            entries: IndexMap::new(),
        }
    }

    /// Add an entry, keeping insertion order.
    pub fn insert(
        &mut self,
        key: K,
        value: NestedValue<K, V>,
    ) where
        K: std::hash::Hash + Eq,
    {
        self.entries.insert(key, value);
    }
}

impl<K, V> NestedMap<K, V>
where
    K: Display,
{
    /// Flatten the whole tree into `.`-joined keys, in insertion order.
    pub fn flatten(self) -> IndexMap<String, V> {
        NestedValue::Map(self.entries).flatten("")
    }
}

impl<K, V> From<NestedMap<K, V>> for NestedValue<K, V> {
    fn from(map: NestedMap<K, V>) -> Self {
        NestedValue::Map(map.entries)
    }
}
