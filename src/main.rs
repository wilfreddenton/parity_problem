mod network;
mod util;
use std::io;
use std::io::prelude::*;
use std::process::exit;
use network::Network;
use util::{generate_input_data, expected_output, output_error, etas};

fn train_network(network: &mut Network, input_data: &Vec<Vec<f32>>, tolerance: f32, eta: f32) -> (u64, Vec<f32>) {
    let mut num_epochs = 0;
    let mut converged: bool = true;
    loop {
        num_epochs = num_epochs + 1;
        for input in input_data.iter() {
            let outputs: Vec<Vec<f32>> = network.feed(&input).unwrap();
            let expected_outputs: Vec<f32> = vec!(expected_output(&input));
            network.backpropagate(&input, &outputs, &expected_outputs, eta);
            let output_layer_outputs: &Vec<f32> = outputs.last().unwrap();
            if converged == true && tolerance < output_error(&output_layer_outputs, &expected_outputs) {
                // println!("{:?}\t{}", input, output_error(&output_layer_outputs, &expected_outputs));
                converged = false;
            }
        }
        if converged {
            break;
        }
        converged = true;
    }
    let mut results: Vec<f32> = Vec::new();
    for input in input_data.iter() {
        let outputs: Vec<Vec<f32>> = network.feed(&input).unwrap();
        results.push(outputs[outputs.len() - 1][0]);
    }
    (num_epochs, results)
}

fn main() {
    let etas: Vec<f32> = etas();
    let num_neurons_in_layers = [4, 1];
    let num_inputs = 4;
    let input_data = generate_input_data();
    let mut stdin = io::stdin();
    let mut selection = String::new();
    println!("\nTrain a network to solve the parity problem!");
    println!("============================================");
    println!("1) train with specific eta");
    println!("2) train with a range of etas");
    println!("3) train with a range of etas and with momentum alpha = 0.9");
    print!("#: ");
    io::stdout().flush();
    stdin.read_line(&mut selection).unwrap();
    let s = selection.trim().parse::<u8>().unwrap();
    let mut training_results: Vec<(u64, Vec<f32>)> = Vec::new();
    match s {
        1 => {
            print!("value for eta: ");
            io::stdout().flush();
            let mut eta: String = String::new();
            stdin.read_line(&mut eta).unwrap();
            let e = eta.trim().parse::<f32>().unwrap();
            let mut network = Network::new(&num_neurons_in_layers, num_inputs);
            println!("training network with eta: {}...", e);
            let (num_epochs, result) = train_network(&mut network, &input_data, 0.05, e);
            training_results.push((num_epochs, result));
            println!("finished network with eta: {}, # epochs = {}", e, num_epochs);
        },
        2 => {
            for e in etas.iter() {
                let mut network = Network::new(&num_neurons_in_layers, num_inputs);
                println!("training network with eta: {}...", e);
                let (num_epochs, result) = train_network(&mut network, &input_data, 0.05, *e);
                training_results.push((num_epochs, result));
                println!("finished network with eta: {}, # epochs = {}", e, num_epochs);
            }
        },
        3 => {
        },
        _ => {
            println!("You did not enter a valide option, exiting...");
            exit(0);
        }
    }
    for (eta, results) in etas.iter().zip(training_results.iter()) {
        let (num_epochs, result) = results.to_owned();
        println!("# epochs for eta = {}: {}", eta, num_epochs);
        for (input, output) in input_data.iter().zip(result.iter()) {
            println!("For input {:?}: output = {}, expected output = {}", input, output, expected_output(&input));
        }
    }
}

