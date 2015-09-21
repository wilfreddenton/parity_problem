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
    pub fn signal_neurons(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs: Vec<f64> = Vec::new();
        for n in self.neurons.iter() {
            outputs.push(n.signal(inputs));
        }
        outputs
    }
}
