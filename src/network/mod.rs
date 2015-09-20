pub mod network_layer;
use network::network_layer::NetworkLayer;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<NetworkLayer>,
}

impl Network {
    pub fn new(alpha: f64, num_neurons_in_layer: &[usize], num_inputs: usize) -> Network {
        let inputs_arr: [usize; 2] = [num_inputs, num_neurons_in_layer[0]];
        let mut layers: Vec<NetworkLayer> = Vec::new();
        for (l, i) in num_neurons_in_layer.iter().zip(inputs_arr.iter()) {
            layers.push(NetworkLayer::new(alpha, *l, *i));
        }
        Network {
            layers: layers,
        }
    }
    pub fn output(inputs: Vec<f64>) -> f64 {
        0.3
    }
    pub fn pretty_print(network: &Network) {
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
}

