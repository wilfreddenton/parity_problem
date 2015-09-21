mod network;
use network::Network;

fn permutations(perms: &mut Vec<Vec<f64>>, perm: &mut Vec<f64>, elements: &[f64], n: usize) {
    if n == 0 {
        let perm = perm.to_owned();
        perms.push(perm);
        return;
    }
    for e in elements.iter() {
        let mut perm_cp = perm.to_owned();
        perm_cp.push(*e);
        permutations(perms, &mut perm_cp, elements, n - 1);
    }
}

fn generate_input_data() -> Vec<Vec<f64>> {
    let mut perms: Vec<Vec<f64>> = Vec::new();
    let mut perm: Vec<f64> = Vec::new();
    let n: usize = 4;
    let elements = [0.0, 1.0];
    permutations(&mut perms, &mut perm, &elements, n);
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
    let num_neurons_in_layers = [4, 1];
    let num_inputs = 4;
    let network = Network::new(&num_neurons_in_layers, num_inputs);
    let input_data = generate_input_data();
    let target = expected_output(&input_data[14]);
    let actual = network.feed(&input_data[14]);
    println!("target: {}, actual: {:?}", target, actual);
    //println!("output: {}", output);
    //println!("input: {:?}", input_data);
    //for i in input_data.iter() {
    //    println!("input: {:?}", i);
    //    let output = expected_output(&i);
    //    println!("output: {}", output);
    //}
}

