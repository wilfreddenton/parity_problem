extern crate rand;
use std::f32::consts::E;
use self::rand::distributions::{IndependentSample, Range};

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f32>,
}

impl Neuron {
    fn activation_potential(&self, inputs: &Vec<f32>) -> f32 {
        let mut inputs: Vec<f32> = inputs.to_owned();
        let mut v = 0.0;
        // add an input for the bias
        inputs.insert(0, 1.0);
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            v = v + w * x;
        }
        v
    }
    fn activation_function(&self, activation_potential: f32) -> f32 {
        1.0 / (1.0 + E.powf(-activation_potential))
    }
    pub fn new(num_inputs: usize) -> Neuron {
        let between = Range::new(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        let mut weights: Vec<f32> = Vec::new();
        // add 1 to include the bias as w0
        for _ in 0..(num_inputs + 1) {
            weights.push(between.ind_sample(&mut rng));
        }
        Neuron {
            weights: weights,
        }
    }
    pub fn update_weights(&mut self, delta_weights: &Vec<f32>) {
        let mut weights: Vec<f32> = Vec::new();
        for (dw, w) in delta_weights.iter().zip(self.weights.iter()) {
            weights.push(*w + *dw);
        }
        self.weights = weights;
    }
    pub fn signal(&self, inputs: &Vec<f32>) -> f32 {
        let v = self.activation_potential(inputs);
        let phi_prime: f32 = self.activation_function(v);
        phi_prime
    }
}

