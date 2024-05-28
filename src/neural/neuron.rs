use core::fmt;

use std::{error::Error, f64};

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
    pub fn activate<F>(
        &self,
        input: &Vec<f64>,
        f: F,
    ) -> Result<f64, InputWeightLengthsMismatchError>
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
#[cfg(test)]
mod tests {

    use super::*;
    use crate::{ActivationF, Identity};
    #[test]
    fn should_return_error() -> Result<(), String> {
        let n = Neuron {
            weights: vec![],
            bias: 1.,
        };
        let r = n.activate(&vec![1.], |x| Identity.f(x));
        match r {
            Ok(_) => Err(String::from(
                "Should error with InputWeightLengthsMismatchError",
            )),
            Err(_) => Ok(()),
        }
    }

    #[test]
    fn should_not_return_error() -> Result<(), InputWeightLengthsMismatchError> {
        let n = Neuron {
            weights: vec![],
            bias: 1.,
        };
        let r = n.activate(&vec![], |x| Identity.f(x));
        match r {
            Ok(_) => Ok(()),
            Err(err) => Err(err),
        }
    }

    #[test]
    fn should_add_bias() -> Result<(), InputWeightLengthsMismatchError> {
        let n = Neuron {
            weights: vec![],
            bias: 1.,
        };
        let r = n.activate(&vec![], |x| Identity.f(x))?;
        assert_eq!(r, 1.);
        Ok(())
    }

    #[test]
    fn should_sum_inputs() -> Result<(), InputWeightLengthsMismatchError> {
        let len = 10;
        let n = Neuron {
            weights: vec![1.; len],
            bias: 0.,
        };
        let r = n.activate(&vec![1.; len], |x| Identity.f(x))?;
        assert_eq!(r, 10.);
        Ok(())
    }
}
