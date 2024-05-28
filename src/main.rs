use crate::neural::activation_funcs::{ActivationF, Relu};
use crate::neural::neuron::Neuron;

pub mod neural;

fn main() {
    let a = Neuron {
        weights: vec![1., 2.],
        bias: 1.,
    };

    let relu = Relu {};
    println!(
        "Hello, neuron: {:?}!",
        a.activate(vec![1., 1.], |x| relu.f(x))
    );
}
