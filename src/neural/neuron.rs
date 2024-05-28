use core::fmt;

use std::{error::Error, f64, result::Result};

#[derive(Debug)]
pub struct Neuron {
    pub(crate) weights: Vec<f64>,
    pub(crate) bias: f64,
}
#[derive(Debug)]
pub struct InputWeightLengthsMismatchError {
    inputs_lengt: usize,
    weights_lenght: usize,
}

impl Error for InputWeightLengthsMismatchError {}

impl fmt::Display for InputWeightLengthsMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Inputs lengths and weights length is not the same, inputs lengts :{}, weights length: {}", self.inputs_lengt, self.weights_lenght)
    }
}
impl Neuron {
    pub fn activate<F>(&self, input: Vec<f64>, f: F) -> Result<f64, InputWeightLengthsMismatchError>
    where
        F: Fn(f64) -> f64,
    {
        if input.len() != self.weights.len() {
            return Err(InputWeightLengthsMismatchError {
                inputs_lengt: input.len(),
                weights_lenght: self.weights.len(),
            });
        }
        let mut sum = self.bias;

        for i in 0..input.len() {
            sum += self.weights[i] * input[i]
        }
        let output = f(sum);
        return Ok(output);
    }
}
