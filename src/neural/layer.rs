use std::{
    error::{self, Error},
    fmt,
};

use super::activation_funcs::ActivationFunc;

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
impl error::Error for LayerError {}

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
        write!(
            f,
            "Error while activating neuron: {}, source: {:?}",
            self.neuron_index, self.source
        )
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

pub struct Layer {
    pub neurons: Vec<Neuron>,
    layer_type: Types,
    f: ActivationFunc,
}

impl Layer {
    pub fn new<F>(
        num_neuron: usize,
        num_inputs: usize,
        layer_type: Types,
        f: ActivationFunc,
        mut initializer: F,
    ) -> Result<Self, LayerError>
    where
        F: FnMut() -> f64,
    {
        if num_neuron != num_inputs && matches!(layer_type, Types::OneOnOne) {
            return Err(LayerError::InputOneOnOneError(InputOneOnOneError {
                num_inputs,
                expected_num_inputs: num_neuron,
            }));
        }
        let neurons = (0..num_neuron)
            .map(|_| Neuron {
                weights: match layer_type {
                    Types::OneOnOne => {
                        vec![initializer()]
                    }
                    Types::Fully => (0..num_inputs).map(|_| initializer()).collect(),
                },
                bias: initializer(),
            })
            .collect();
        Ok(Self {
            neurons,
            layer_type,
            f,
        })
    }

    pub fn input(&self, xs: &Vec<f64>) -> Result<Vec<f64>, LayerError> {
        return self
            .neurons
            .iter()
            .enumerate()
            .map(|(i, x)| match self.layer_type {
                Types::OneOnOne => match x.activate(&vec![xs[i]], |t| self.f.0(t)) {
                    Ok(n) => Ok(n),
                    Err(e) => Err(LayerError::NeuronActivationError(NeuronActivationError {
                        source: Some(e),
                        neuron_index: i,
                    })),
                },
                Types::Fully => match x.activate(&xs, |t| self.f.0(t)) {
                    Ok(n) => Ok(n),
                    Err(e) => Err(LayerError::NeuronActivationError(NeuronActivationError {
                        source: Some(e),
                        neuron_index: i,
                    })),
                },
            })
            .collect();
    }
    pub fn count_of_weights(&self) -> usize {
        self.neurons
            .iter()
            .fold(0, |sum, neuron| sum + neuron.weights_length())
    }
}
