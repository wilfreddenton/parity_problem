pub mod network_layer;
use network::network_layer::NetworkLayer;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<NetworkLayer>,
    pub num_inputs: usize,
}

impl Network {
    pub fn new(num_neurons_in_layer: &[usize], num_inputs: usize) -> Network {
        let inputs_arr: [usize; 2] = [num_inputs, num_neurons_in_layer[0]];
        let mut layers: Vec<NetworkLayer> = Vec::new();
        for (l, i) in num_neurons_in_layer.iter().zip(inputs_arr.iter()) {
            layers.push(NetworkLayer::new(*l, *i));
        }
        Network {
            layers: layers,
            num_inputs: num_inputs,
        }
    }
    pub fn feed(&self, inputs: &Vec<f64>) -> Result<Vec<f64>, String> {
        if inputs.len() != self.num_inputs {
            return Err(format!("This network has been configured to consume {} inputs. An array with {} elements was fed.", self.num_inputs, inputs.len()));
        }
        let mut inputs = inputs.to_owned();
        for l in self.layers.iter() {
            inputs = l.signal_neurons(&inputs);
        }
        Ok(inputs)
    }
    pub fn pretty_print(network: &Network) {
        for layer in network.layers.iter() {
            println!("LAYER");
            for neuron in layer.neurons.iter() {
                println!("  NEURON");
                println!("    bias: {}", neuron.bias);
                println!("    connections:");
                for w in neuron.weights.iter() {
                    println!("      weight: {}", w);
                }
            }
        }
    }
}

