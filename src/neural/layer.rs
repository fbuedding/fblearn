use std::{error::Error, fmt::{self}};

use crate::ActivationF;

use super::neuron::{InputWeightLengthsMismatchError, Neuron};

#[derive(Debug)]
pub enum LayerError {
    NeuronActivationError(NeuronActivationError),
    InputOneOnOneError(InputOneOnOneError),
}
impl<'a> fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerError::NeuronActivationError(err) => fmt::Display::fmt(&err, f),
            LayerError::InputOneOnOneError(err) => fmt::Display::fmt(&err, f),
        }
    }
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
        write!(f, "Error while activating neuron: {}, source: {:?}", self.neuron_index, self.source)
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

pub struct Layer<'a, T> where T: ActivationF{
    neurons: Vec<Neuron>,
    layer_type: Types,
    num_inputs: usize,
    f: &'a T
}

impl<'a, T:ActivationF> Layer<'a, T> {
    pub fn input(&self, xs: &Vec<f64>) -> Result<Vec<f64>, LayerError> {
        return self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, x)| match self.layer_type {
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

/// .
pub fn new_layer <'a, T : ActivationF, F>(
    num_neuron: usize,
    num_inputs: usize,
    layer_type: Types,
    activation_func: &'a T,
    mut initializer: F
) -> Layer <T> where F: FnMut() ->  f64{

    let neurons = (0..num_neuron)
        .map(|_| Neuron {
            weights: match layer_type {
                Types::OneOnOne => vec![initializer()],
                Types::Fully => (0..num_neuron).map(|_| initializer()).collect(),
            },
            bias: initializer(),
        })
        .collect();

    return Layer {
        neurons,
        num_inputs,
        layer_type,
        f: activation_func,
    };
}
