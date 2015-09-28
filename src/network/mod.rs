pub mod network_layer;
use network::network_layer::NetworkLayer;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<NetworkLayer>,
    pub num_inputs: usize,
}

impl Network {
    pub fn new(num_neurons_in_layers: &[usize], num_inputs: usize) -> Network {
        let inputs_arr: [usize; 2] = [num_inputs, num_neurons_in_layers[0]];
        let mut layers: Vec<NetworkLayer> = Vec::new();
        for (l, i) in num_neurons_in_layers.iter().zip(inputs_arr.iter()) {
            layers.push(NetworkLayer::new(*l, *i));
        }
        Network {
            layers: layers,
            num_inputs: num_inputs,
        }
    }
    pub fn feed(&self, inputs: &Vec<f32>) -> Result<Vec<Vec<f32>>, String> {
        if inputs.len() != self.num_inputs {
            return Err(format!("This network has been configured to consume {} inputs. An array with {} elements was fed.", self.num_inputs, inputs.len()));
        }
        let mut inputs = inputs.to_owned();
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        for l in self.layers.iter() {
            inputs = l.signal_neurons(&inputs);
            outputs.push(inputs.to_owned());
        }
        Ok(outputs)
    }
    pub fn update_weights(&mut self, delta_weights: &Vec<Vec<Vec<f32>>>) {
        for (dw, l) in delta_weights.iter().zip(self.layers.iter_mut()) {
           l.update_weights(dw);
        }
    }
    pub fn backpropagate(&mut self, inputs: &Vec<f32>, outputs: &Vec<Vec<f32>>, expected_outputs: &Vec<f32>, eta: f32) {
        let phi_prime = |phi: f32| -> f32 {
            // no a because a is 1.0
            phi * (1.0 - phi)
        };
        let delta_k = |d: f32, y: f32| -> f32 {
            (d - y) * phi_prime(y)
        };
        let delta_j = |y: f32, gradients: &Vec<f32>, weights: &Vec<f32>| -> f32 {
            let sum: f32 = gradients.iter().zip(weights.iter())
                .map(|(g, w)| *g * *w)
                .fold(0.0, |sum, val| sum + val);
            phi_prime(y) * sum
        };
        // instantiate the new weights vector. it's 3d because (layer(neuron(weight
        let mut delta_weights: Vec<Vec<Vec<f32>>> = Vec::new();
        // grab a copy of the input vector
        let inputs: Vec<f32> = inputs.to_owned();
        // grab a copy of the output vector. it's 2d because (layer(output
        let mut outputs: Vec<Vec<f32>> = outputs.to_owned();
        // add the inputs as the first elements in the outputs vector. this is done
        // so we can loop over the outputs and get the input layer in the iteration
        outputs.insert(0, inputs);
        // add bias input in each of the layer outputs except the global output
        for i in 0..(outputs.len() - 1) {
            outputs[i].insert(0, 1.0);
        }
        // get the global output from the outputs vector
        let k_outputs: &Vec<f32> = outputs.last().unwrap();
        // create a vector to hold all the weights in the output layer. it's 2d
        // because (neuron(weight updates
        let mut delta_w_k: Vec<Vec<f32>> = Vec::new();
        // create a vector to hold the "collecting" gradients.
        let mut gradients: Vec<f32> = Vec::new();
        // loop over the outputs of the output layer and their expected outputs
        for (y, d) in k_outputs.iter().zip(expected_outputs.iter()) {
            // create a vector to hold the change of all the weights in neuron j
            let mut delta_w_kj: Vec<f32> = Vec::new();
            // create a vector to hold the gradient of Wkj
            let gradient = delta_k(*d, *y);
            // push the gradient into the gradients for later use
            gradients.push(gradient);
            // for each output of the last (we're going backwards) hidden layer
            for yj in outputs.get(outputs.len() - 2).unwrap().iter() {
                // find the change in weight for this input and add it to the
                // delta_kj vector
                delta_w_kj.push(eta * gradient * yj);
            }
            // add this neurons weight updates to the delta_w_k vector
            delta_w_k.push(delta_w_kj);
        }
        // add the output layer's neuron's weights to the global weight change vector
        delta_weights.push(delta_w_k);
        // loop over the remaining outputs and layers in reverse order since we are
        // backpropagating
        // the -1 is NOT for the index, rust ranges are exclusive at the end. the -1
        // is to skip the output layer. the range ends at 1 because at index 0 is
        // the input layer
        for layer_index in (1..(outputs.len() - 1)).rev() {
            // create a vector to point to the next layer
            let layer_k: &NetworkLayer = &self.layers[layer_index];
            // create a vector to point to the inputs (outputs of the previous layer)
            let layer_inputs: &Vec<f32> = &outputs[layer_index - 1];
            // create a vector to point to this hidden layer's outputs
            let layer_outputs: &Vec<f32> = &outputs[layer_index];
            // create a vector to hold the weight updates of hidden layer j. it's 2d
            // because (neuron(weight updates
            let mut delta_w_j: Vec<Vec<f32>> = Vec::new();
            // create a vector to hold the new gradients
            let mut new_gradients: Vec<f32> = Vec::new();
            // loop through the outputs of the hidden layer. remember that the output
            // at index 0 is the bias
            for j in 0..layer_outputs.len() {
                // create a &f32 to point to the jth neuron's output in the hidden
                // layer
                let yj: &f32 = &layer_outputs[j];
                // create a vector to hold the weight updates of the neuron
                let mut delta_w_ji: Vec<f32> = Vec::new();
                // get wkj which is a vector of each j weight in each k neuron
                let mut w_kj: Vec<f32> = Vec::new();
                // loop over each neuron in next layer
                for k in 0..layer_k.neurons.len() {
                    // loop over each weight in the kth neuron of the layer
                    for j_weight_index in 0..layer_k.neurons[k].weights.len() {
                        if j == j_weight_index {
                            w_kj.push(layer_k.neurons[k].weights[j_weight_index]);
                        }
                    }
                }
                // compute the gradient of the weight
                let gradient: f32 = delta_j(*yj, &gradients, &w_kj);
                // add the gradient to the new_gradients vector
                new_gradients.push(gradient);
                // loop through all the inputs
                for yi in layer_inputs.iter() {
                    // push the weight update into the vector delta_w_ji
                    delta_w_ji.push(eta * gradient * yi);
                }
                // add the weight updates for the neuron to the layer weight updates
                // vector. skip the first because it's the bias
                if j != 0 {
                    delta_w_j.push(delta_w_ji);
                }
            }
            // set the current gradients to the new gradients
            gradients = new_gradients;
            // add the weight updates for the layer to the global weight updates
            // vector
            delta_weights.insert(0, delta_w_j);
        }
        // println!("{:?}", delta_weights);
        // batch update the weights
        self.update_weights(&delta_weights);
    }
    pub fn pretty_print(network: &Network) {
        for layer in network.layers.iter() {
            println!("LAYER");
            for neuron in layer.neurons.iter() {
                println!("  NEURON");
                println!("    connections:");
                for w in neuron.weights.iter() {
                    println!("      weight: {}", w);
                }
            }
        }
    }
}

// tests

#[cfg(test)]
mod tests {
    #[test]
    fn test_new_network() {
        let num_neurons_in_layers: [usize; 2] = [4, 1];
        let num_inputs: usize = 4;
        let network = super::Network::new(&num_neurons_in_layers, num_inputs);
        assert_eq!(num_neurons_in_layers.len(), network.layers.len());
        for (num_neurons, layer) in num_neurons_in_layers.iter().zip(network.layers.iter()) {
            assert_eq!(num_neurons.to_owned(), layer.neurons.len());
        }
        assert_eq!(num_inputs, network.num_inputs);
    }
    #[test]
    fn test_update_weights() {
        // use 3 delta weights because the bias is w0
        let hidden_layer_delta_weights: Vec<Vec<f32>> = vec!(vec!(0.1, 0.2, 0.3), vec!(0.4, 0.5, 0.6));
        let output_layer_delta_weights: Vec<Vec<f32>> = vec!(vec!(0.1, 0.2, 0.3), vec!(0.4, 0.5, 0.6));
        let delta_weights = vec!(hidden_layer_delta_weights, output_layer_delta_weights);
        let num_neurons_in_layers: [usize; 2] = [2, 2];
        let num_inputs: usize = 2;
        let mut network = super::Network::new(&num_neurons_in_layers, num_inputs);
        let mut expected_weights: Vec<f32> = Vec::new();
        let mut old_weights: Vec<f32> = Vec::new();
        for (l, dw) in network.layers.iter().zip(delta_weights.iter()) {
            for (n, dw) in l.neurons.iter().zip(dw.iter()) {
                for (w, dw) in n.weights.iter().zip(dw.iter()) {
                    old_weights.push(*w);
                    expected_weights.push(*w + *dw);
                }
            }
        }
        network.update_weights(&delta_weights);
        let mut actual_weights: Vec<f32> = Vec::new();
        for l in network.layers.iter() {
            for n in l.neurons.iter() {
                for w in n.weights.iter() {
                    actual_weights.push(*w);
                }
            }
        }
        for ((ew, aw), ow) in expected_weights.iter().zip(actual_weights.iter()).zip(old_weights.iter()) {
            println!("{}: {}, {}", *ow, *ew, *aw);
            assert_eq!(*ew, *aw);
        }
    }
}
