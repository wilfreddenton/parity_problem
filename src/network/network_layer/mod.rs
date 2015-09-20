pub mod neuron;
use network::network_layer::neuron::Neuron;

#[derive(Debug)]
pub struct NetworkLayer {
    pub neurons: Vec<Neuron>,
}

impl NetworkLayer {
    pub fn new(alpha: f64, num_neurons: usize, num_inputs: usize) -> NetworkLayer {
        let mut neurons: Vec<Neuron> = Vec::new();
        for _ in 0..num_neurons {
            neurons.push(Neuron::new(alpha, num_inputs));
        }
        NetworkLayer {
            neurons: neurons,
        }
    }
}
