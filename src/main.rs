use rand::Rng;

use crate::neural::activation_funcs::*;
use crate::neural::layer::new_layer;

pub mod neural;

fn main() {
    let relu = Relu {};

    let mut rng = rand::thread_rng();
    let layer = new_layer(100, 1, neural::layer::Types::Fully, &relu, || rng.gen());
    let out = layer.input(&vec![1.;100]);
        match out {
            Ok(v) => println!("{:?}", v),
            Err(err) => eprintln!("{err}"),
        }
}
