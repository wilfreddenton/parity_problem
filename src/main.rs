mod network;
use network::Network;

fn permutations(perms: &mut Vec<Vec<f64>>, perm: &mut Vec<f64>, data: &[f64], n: u32) {
    if n == 0 {
        let perm = perm.to_owned();
        perms.push(perm);
        return;
    }
    for d in data.iter() {
        let mut perm_cp = perm.to_owned();
        perm_cp.push(*d);
        permutations(perms, &mut perm_cp, data, n - 1);
    }
}

fn generate_input_data() -> Vec<Vec<f64>> {
    let mut perms: Vec<Vec<f64>> = Vec::new();
    let mut perm: Vec<f64> = Vec::new();
    let n: u32 = 4;
    let data = [0.0, 1.0];
    permutations(&mut perms, &mut perm, &data, n);
    perms
}

fn expected_output(input: &Vec<f64>) -> f64 {
    let num_ones: usize = input.iter()
        .filter(|&i| *i == 1.0)
        .collect::<Vec<_>>()
        .len();
    if num_ones % 2 == 0 {
        0.0
    } else {
        1.0
    }
}

fn main() {
    let alpha = 1.0;
    let num_neurons_in_layers = [4, 1];
    let num_inputs = 4;
    let network = Network::new(alpha, &num_neurons_in_layers, num_inputs);
    let input_data = generate_input_data();
    let output = network.layers[0].neurons[0].signal(vec![1.0, 1.0, 0.0, 0.0]);
    println!("output: {}", output);
    println!("input: {:?}", input_data);
    for i in input_data.iter() {
        println!("input: {:?}", i);
        let output = expected_output(&i);
        println!("output: {}", output);
    }
}

