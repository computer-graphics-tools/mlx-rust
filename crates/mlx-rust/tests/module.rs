//! The parameter tree the whole `nn` layer rests on: declaration-order
//! flattening, freezing, replacement, and safetensors round-trips.

use mlx::{
    Array,
    module::{Module, ModuleParameters, Param},
    ops,
};

/// Shaped like the real `Linear` so the test exercises what layers will do:
/// a required parameter, an optional one, and a non-parameter config field.
#[derive(Debug, ModuleParameters)]
struct Linear {
    #[param]
    weight: Param<Array>,
    #[param]
    bias: Param<Option<Array>>,
    in_features: i32,
}

impl Linear {
    fn new(with_bias: bool) -> Self {
        Linear {
            weight: Param::new(
                Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
            ),
            bias: Param::new(
                with_bias
                    .then(|| Array::from_slice(&[0.5f32, 1.5], &[2]).unwrap()),
            ),
            in_features: 2,
        }
    }
}

impl Module<&Array> for Linear {
    type Output = Array;
    type Error = mlx::Error;

    fn forward(
        &mut self,
        input: &Array,
    ) -> mlx::Result<Array> {
        let output = ops::matmul(input, &*self.weight)?;
        match self.bias.as_ref() {
            Some(bias) => ops::add(&output, bias),
            None => Ok(output),
        }
    }

    fn training_mode(
        &mut self,
        _mode: bool,
    ) {
    }
}

/// A module holding a module, so the tree is actually nested.
#[derive(Debug, ModuleParameters)]
struct TwoLayer {
    #[param]
    first: Linear,
    #[param]
    second: Linear,
}

#[test]
fn flatten_names_parameters_by_path_in_declaration_order() {
    let model = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(false),
    };

    // A field without `#[param]` is configuration, not a parameter: it must not
    // appear in the tree, and must still be readable.
    assert_eq!(model.first.in_features, 2);

    let keys: Vec<String> = model.flatten().keys().cloned().collect();
    assert_eq!(
        keys,
        vec![
            "first.weight".to_string(),
            "first.bias".to_string(),
            "second.weight".to_string(),
        ],
        "an absent optional parameter must not appear, and order must follow \
         declaration"
    );
    assert_eq!(model.num_parameters(), 3);
}

/// The `indexmap` claim: ordering must not vary run to run, or checkpoints and
/// optimizer state are not reproducible. `HashMap` would fail this.
#[test]
fn parameter_order_is_stable_across_many_rebuilds() {
    let expected: Vec<String> = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(true),
    }
    .flatten()
    .keys()
    .cloned()
    .collect();

    for round in 0..100 {
        let keys: Vec<String> = TwoLayer {
            first: Linear::new(true),
            second: Linear::new(true),
        }
        .flatten()
        .keys()
        .cloned()
        .collect();
        assert_eq!(keys, expected, "ordering changed on round {round}");
    }
}

#[test]
fn update_flattened_replaces_only_matching_keys() {
    let mut model = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(true),
    };

    let mut replacements = indexmap::IndexMap::new();
    replacements.insert(
        "first.weight".to_string(),
        Array::from_slice(&[9.0f32; 4], &[2, 2]).unwrap(),
    );
    // A key no parameter matches must be ignored, not panic: checkpoints carry
    // extra entries.
    replacements.insert(
        "nonexistent".to_string(),
        Array::from_slice(&[0.0f32], &[1]).unwrap(),
    );
    model.update_flattened(&replacements);

    assert_eq!(model.first.weight.to_vec_f32().unwrap(), vec![9.0; 4]);
    assert_eq!(
        model.second.weight.to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0],
        "the other layer must be untouched"
    );
}

#[test]
fn freezing_removes_parameters_from_the_trainable_set() {
    let mut model = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(true),
    };
    assert_eq!(model.trainable_parameters().flatten().len(), 4);
    assert_eq!(model.all_frozen(), Some(false));

    model.first.freeze_parameters(true);
    let trainable: Vec<String> =
        model.trainable_parameters().flatten().keys().cloned().collect();
    assert_eq!(
        trainable,
        vec!["second.weight".to_string(), "second.bias".to_string()],
        "frozen parameters must drop out of the trainable set"
    );
    assert_eq!(model.any_frozen(), Some(true));

    model.first.unfreeze_parameters(true);
    assert_eq!(model.trainable_parameters().flatten().len(), 4);
}

#[test]
fn safetensors_round_trip_restores_every_parameter() {
    let directory = std::env::temp_dir().join("mlx_rust_module_test");
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("model.safetensors");

    let saved = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(true),
    };
    saved.save_safetensors(&file).unwrap();

    // Load into a model whose weights differ, so a no-op would be visible.
    let mut loaded = TwoLayer {
        first: Linear::new(true),
        second: Linear::new(true),
    };
    loaded.first.weight =
        Param::new(Array::from_slice(&[0.0f32; 4], &[2, 2]).unwrap());
    loaded.load_safetensors(&file).unwrap();

    for (key, expected) in saved.flatten() {
        let restored = loaded.flatten();
        assert_eq!(
            restored[&key].to_vec_f32().unwrap(),
            expected.to_vec_f32().unwrap(),
            "{key} did not survive the round-trip"
        );
    }

    std::fs::remove_dir_all(&directory).ok();
}

#[test]
fn forward_runs_through_a_derived_module() {
    let mut layer = Linear::new(true);
    let input = Array::from_slice(&[1.0f32, 1.0], &[1, 2]).unwrap();
    // [1,1] @ [[1,2],[3,4]] = [4,6]; plus bias [0.5,1.5] = [4.5,7.5]
    assert_eq!(
        layer.forward(&input).unwrap().to_vec_f32().unwrap(),
        vec![4.5, 7.5]
    );
}
