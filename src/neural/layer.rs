use std::{error::Error, fmt};

use crate::ActivationF;

use super::neuron::{InputWeightLengthsMismatchError, Neuron};
use rand::prelude::*;

#[derive(Debug)]
pub enum LayerError {
    NeuronActivationError(NeuronActivationError),
    InputOneOnOneError(InputOneOnOneError),
}

#[derive(Debug)]
pub struct InputOneOnOneError {
    num_inputs: usize,
    expected_num_inputs: usize,
}
impl<'a> fmt::Display for InputOneOnOneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "When layer type is OneOnOne the input length has to be the same as the neuron count, num_inputs: {}, expected: {}" ,self.num_inputs, self.expected_num_inputs)
    }
}

#[derive(Debug)]
pub struct NeuronActivationError {
    source: Option<InputWeightLengthsMismatchError>,
    neuron_index: usize,
}

impl<'a> fmt::Display for NeuronActivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error while activating neuron: {}", self.neuron_index)
    }
}

impl<'a> Error for NeuronActivationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            Some(err) => Some(err),
            None => None,
        }
    }
}

pub enum Types {
    OneOnOne,
    Fully,
}

pub struct Layer<'a> {
    neurons: Vec<Neuron>,
    connection: Types,
    num_inputs: usize,
    f: &'a dyn ActivationF,
}

impl<'a> Layer<'a> {
    pub fn input(&self, xs: Vec<f64>) -> Result<Vec<f64>, LayerError> {
        return self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, x)| match self.connection {
                Types::OneOnOne => {
                    if xs.len() != self.num_inputs {
                        return Err(LayerError::InputOneOnOneError(InputOneOnOneError {
                            num_inputs: xs.len(),
                            expected_num_inputs: self.num_inputs,
                        }));
                    }

                    match x.activate(&vec![xs[i]], |t| self.f.f(t)) {
                        Ok(n) => Ok(n),
                        Err(e) => Err(LayerError::NeuronActivationError(NeuronActivationError {
                            source: Some(e),
                            neuron_index: i,
                        })),
                    }
                }
                Types::Fully => match x.activate(&xs, |t| self.f.f(t)) {
                    Ok(n) => Ok(n),
                    Err(e) => Err(LayerError::NeuronActivationError(NeuronActivationError {
                        source: Some(e),
                        neuron_index: i,
                    })),
                },
            })
            .collect();
    }
}

pub fn new_layer(
    num_neuron: usize,
    num_inputs: usize,
    connection: Types,
    f: &dyn ActivationF,
) -> Layer {
    let mut rng = rand::thread_rng();

    let neurons = (0..num_neuron)
        .map(|_| Neuron {
            weights: match connection {
                Types::OneOnOne => vec![rng.gen()],
                Types::Fully => (0..num_inputs).map(|_| rng.gen()).collect(),
            },
            bias: rng.gen(),
        })
        .collect();

    return Layer {
        neurons,
        num_inputs,
        connection,
        f,
    };
}
