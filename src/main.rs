mod network;
mod util;
use network::Network;
use util::{generate_input_data, expected_output, output_error};

fn main() {
    let eta = 0.05;
    let mut num_epochs = 0;
    let num_neurons_in_layers = [4, 1];
    let num_inputs = 4;
    let input_data = generate_input_data();
    println!("{:?}", input_data);
    let mut network = Network::new(&num_neurons_in_layers, num_inputs);
    let mut converged: bool = true;
    loop {
        num_epochs = num_epochs + 1;
        for input in input_data.iter() {
            let mut outputs: Vec<Vec<f32>> = network.feed(&input).unwrap();
            let expected_outputs: Vec<f32> = vec!(expected_output(&input));
            network.backpropagate(&input, &outputs, &expected_outputs, eta);
            let output_layer_outputs: Vec<f32> = outputs.pop().unwrap();
            if converged == true && 0.05 < output_error(&output_layer_outputs, &expected_outputs) {
                println!("{:?}\t{}", input, output_error(&output_layer_outputs, &expected_outputs));
                converged = false;
            }
        }
        if converged {
            break;
        }
        converged = true;
    }
    for input in input_data.iter() {
        let outputs: Vec<Vec<f32>> = network.feed(&input).unwrap();
        println!("Output: {:?}\tExpected Output: {:?}", outputs[outputs.len() - 1], expected_output(&input));
    }
}

