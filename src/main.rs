use std::error::Error;

use neural::{activation_funcs::SIGMOID, layer::Types::Fully, net::*};

pub mod neural;

fn main() -> Result<(), Box<dyn Error>> {
    let nn = FNN::new(2, Fully, SIGMOID).add_layer(1, Fully, SIGMOID)?;
    println!("Parameter count: {}", nn.parameter_count());
    println!("{:?}", nn.input(vec![1., 2.]));
    Ok(())
}
