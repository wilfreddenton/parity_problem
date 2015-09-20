mod neuron;
use network::neuron::Neuron;

#[derive(Debug)]
pub struct NetworkLayer {
    pub neurons: Vec<Neuron>,
}

