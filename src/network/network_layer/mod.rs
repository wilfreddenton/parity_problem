pub mod neuron;
use network::network_layer::neuron::Neuron;

#[derive(Debug)]
pub struct NetworkLayer {
    pub neurons: Vec<Neuron>,
}

impl NetworkLayer {
    pub fn new(num_neurons: usize, num_inputs: usize) -> NetworkLayer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(num_inputs));
        }
        NetworkLayer {
            neurons: neurons,
        }
    }
    pub fn update_weights(&mut self, delta_weights: &Vec<Vec<f32>>) {
        for (dw, n) in delta_weights.iter().zip(self.neurons.iter_mut()) {
           n.update_weights(dw);
        }
    }
    pub fn signal_neurons(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut outputs: Vec<f32> = Vec::new();
        for n in self.neurons.iter() {
            outputs.push(n.signal(inputs));
        }
        outputs
    }
}
