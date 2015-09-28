fn permutations(perms: &mut Vec<Vec<f32>>, perm: &mut Vec<f32>, elements: &[f32], n: usize) {
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

pub fn generate_input_data() -> Vec<Vec<f32>> {
    let mut perms: Vec<Vec<f32>> = Vec::new();
    let mut perm: Vec<f32> = Vec::new();
    let n: usize = 4;
    let elements = [0.0, 1.0];
    permutations(&mut perms, &mut perm, &elements, n);
    perms
}

pub fn expected_output(input: &Vec<f32>) -> f32 {
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

pub fn output_error(layer_outputs: &Vec<f32>, expected_outputs: &Vec<f32>) -> f32 {
    let mut sum_e_sqrd = 0.0;
    for (d, y) in expected_outputs.iter().zip(layer_outputs.iter()) {
        sum_e_sqrd = sum_e_sqrd + (d - y).powf(2.0);
    }
    0.5 * sum_e_sqrd
}

pub fn phi_prime(a: f32, phi: f32) -> f32 {
    a * phi * (1.0 - phi)
}

// tests

#[cfg(test)]
mod tests {
    #[test]
    fn test_binary_permutations() {
        let mut perms: Vec<Vec<f32>> = Vec::new();
        let expected_perms = [[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[0.0,1.0,1.0],[1.0,0.0,0.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]];
        let mut perm: Vec<f32> = Vec::new();
        let n: usize = 3;
        let elements = [0.0, 1.0];
        super::permutations(&mut perms, &mut perm, &elements, n);
        assert_eq!(expected_perms.len(), perms.len());
        for (ep, p) in expected_perms.iter().zip(perms.iter()) {
            assert_eq!(ep.len(), p.len());
            for (en, n) in ep.iter().zip(p.iter()) {
                assert_eq!(en, n);
            }
        }
    }

    #[test]
    fn test_expected_output() {
        let test_inputs: Vec<Vec<f32>> = vec!(vec!(1.0,0.0,1.0), vec!(1.0, 1.0, 1.0));
        let expected_outputs: Vec<f32> = vec!(0.0, 1.0);
        for (i, eo) in test_inputs.iter().zip(expected_outputs.iter()) {
           let o = super::expected_output(i);
           assert_eq!(eo.to_owned(), o.to_owned());
        }
    }

    #[test]
    fn test_output_error() {
        let layer_outputs: Vec<Vec<f32>> = vec!(vec!(1.0, 2.0, 2.0), vec!(2.0, 3.0, 4.0));
        let expected_outputs: Vec<Vec<f32>> = vec!(vec!(2.0, 3.0, 4.0), vec!(1.0, 2.0, 2.0));
        let correct_errors: Vec<f32> = vec!(3.0, 3.0);
        for ((eo, lo), ce) in expected_outputs.iter().zip(layer_outputs.iter()).zip(correct_errors) {
            assert_eq!(ce, super::output_error(&lo, &eo));
        }
    }
}

