extern crate rand;
use std::f64::consts::E;
use self::rand::distributions::{IndependentSample, Range};

#[derive(Debug)]
pub struct Neuron {
    pub bias: f64,
    pub weights: Vec<f64>,
}

impl Neuron {
    fn activation_potential(&self, inputs: &Vec<f64>) -> f64 {
        let mut v = self.bias;
        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            v = v + w * x;
        }
        v
    }
    fn activation_function(&self, activation_potential: f64) -> f64 {
        let phi = 1.0 / (1.0 + E.powf(-activation_potential));
        let a = 1.0;
        a * phi * (1.0 - phi)
    }
    pub fn new(num_inputs: usize) -> Neuron {
        let between = Range::new(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..num_inputs {
            weights.push(between.ind_sample(&mut rng));
        }
        Neuron {
            bias: between.ind_sample(&mut rng),
            weights: weights,
        }
    }
    pub fn signal(&self, inputs: &Vec<f64>) -> f64 {
        let v = self.activation_potential(inputs);
        let phi_prime: f64 = self.activation_function(v);
        phi_prime
    }
}

