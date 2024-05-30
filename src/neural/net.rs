use std::usize;

use super::activation_funcs::ActivationFunc;
use super::layer::{Layer, LayerError, Types};
use rand::Rng;

// Feeedforward neural network
pub struct FNN {
    layers: Vec<Layer>,
}

impl FNN {
    pub fn new(new_inputs: usize, t: Types, f: ActivationFunc) -> Self {
        let mut rng = rand::thread_rng();
        // Can never error since num_neurons = num_inputs, thus unwrap
        let layers = vec![Layer::new(new_inputs, new_inputs, t, f, || rng.gen()).unwrap()];
        Self { layers }
    }
    pub fn add_layer(
        mut self,
        num_neurons: usize,
        t: Types,
        f: ActivationFunc,
    ) -> Result<Self, LayerError> {
        let num_inputs = match self.layers.last() {
            Some(l) => l.neurons.len(),
            None => num_neurons,
        };
        let mut rng = rand::thread_rng();
        self.layers
            .push(Layer::new(num_neurons, num_inputs, t, f, || rng.gen())?);
        Ok(self)
    }

    pub fn input(&self, inputs: Vec<f64>) -> Result<Vec<f64>, LayerError> {
        self.layers
            .iter()
            .fold(Ok(inputs), |input, layer| layer.input(&input?))
    }
    pub fn parameter_count(&self) -> usize{
       self.layers.iter().fold(0, |sum, layer| sum + layer.count_of_weights())
    }
}



