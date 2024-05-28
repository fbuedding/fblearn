use crate::neural::activation_funcs::*;
use crate::neural::layer::new_layer;

pub mod neural;

fn main() {
    let relu = Relu {};
    let layer = new_layer(1, 1, neural::layer::Types::OneOnOne, &relu);
    println!("Hello, neuron: {:?}!", layer.input(vec![1., 1.]));
}
