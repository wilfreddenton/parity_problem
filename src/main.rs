mod neuron;
use neuron::Neuron;

#[derive(Debug)]
struct NetworkLayer {
    neurons: Vec<Neuron>,
}

#[derive(Debug)]
struct Network {
    layers: Vec<NetworkLayer>,
}

impl Network {
    fn output(inputs: Vec<f64>) -> f64 {
        0.3
    }
}

fn pretty_print_network(network: &Network) {
    for layer in network.layers.iter() {
        println!("LAYER");
        for neuron in layer.neurons.iter() {
            println!("  NEURON");
            println!("    bias: {}, alpha: {}", neuron.bias, neuron.alpha);
            println!("    connections:");
            for w in neuron.weights.iter() {
                println!("      weight: {}", w);
            }
        }
    }
}

fn generate_two_layer_network(alpha: f64) -> Network {
    let mut hidden_neurons: Vec<Neuron> = Vec::new();
    for _ in 0..4 {
        hidden_neurons.push(Neuron::new(alpha, 4));
    }
    let hidden_layer = NetworkLayer {
        neurons: hidden_neurons,
    };
    let output_layer = NetworkLayer {
        neurons: vec![Neuron::new(alpha, 4)],
    };
    Network {
        layers: vec![hidden_layer, output_layer],
    }
}

fn permutations(perms: &mut Vec<Vec<f64>>, perm: &mut Vec<f64>, data: &Vec<f64>, n: u32) {
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
    let data = vec![0.0, 1.0];
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
    let network = generate_two_layer_network(alpha);
    let input_data = generate_input_data();
    let output = network.layers[0].neurons[0].signal(vec![1.0, 1.0, 0.0, 0.0]);
    println!("output: {}", output);
    println!("input: {:?}", input_data);
    for i in input_data.iter() {
        println!("input: {:?}", i);
        let output = expected_output(i);
        println!("output: {}", output);
    }
}

