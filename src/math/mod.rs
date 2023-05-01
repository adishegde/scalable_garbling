use std::cmp::Eq;
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub mod galois;
pub mod utils;

use galois::{GFElement, GFMatrix, GF};

// Compute lagrange coefficients for interpolating a polynomial defined by evaluations at `cpos`
// to evaluations at `npos`.
pub fn lagrange_coeffs(cpos: &[GFElement], npos: &[GFElement], gf: &GF) -> GFMatrix {
    #[cfg(debug_assertions)]
    {
        let num_unique = |v: &[GFElement]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v.dedup();
            v.len()
        };

        // cpos and npos should not have any repetition.
        assert_eq!(num_unique(cpos), cpos.len());
        assert_eq!(num_unique(npos), npos.len());
    }

    let gf_one = gf.one();
    let gf_zero = gf.zero();

    // Pre-compute denominators for all lagrange co-efficients.
    // denom[i] = \prod_{j \neq i} (cpos[i] - cpos[j]).
    let mut denom = Vec::new();
    for i in 0..cpos.len() {
        denom.push(cpos.iter().enumerate().fold(gf_one, |acc, (idx, &x)| {
            if idx == i {
                acc
            } else {
                acc * (cpos[i] - x)
            }
        }));
    }

    let mut all_coeffs = Vec::new();

    for &v in npos {
        // Compute L_1(v), ..., L_n(v) where L_i is the i-th lagrange polynomial.
        let mut coeffs = Vec::new();

        // We first compute the numerator of L_1(v) i.e., (v - cpos[1]) * ... * (v - cpos.last()).
        let mut numerator = cpos.iter().skip(1).fold(gf_one, |acc, &x| acc * (v - x));
        coeffs.push(numerator / denom[0]);

        for j in 1..cpos.len() {
            // Compute L_j(v).
            if numerator != gf_zero {
                numerator *= (v - cpos[j - 1]) / (v - cpos[j]);
                coeffs.push(numerator / denom[j]);
            } else if v == cpos[j] {
                coeffs.push(gf_one);
            } else {
                coeffs.push(gf_zero);
            }
        }

        all_coeffs.push(coeffs);
    }

    all_coeffs
}

pub fn super_inv_matrix(num_inp: usize, num_out: usize, gf: &GF) -> GFMatrix {
    debug_assert!(num_inp >= num_out);

    let mut matrix = Vec::with_capacity(num_out);
    matrix.push(vec![gf.one(); num_inp]);

    for i in 2..(num_out + 1) {
        let val = gf.get(i.try_into().unwrap());
        let mut row = Vec::with_capacity(num_inp);
        row.push(gf.one());
        let mut prev = gf.one();

        for _ in 1..num_inp {
            prev *= val;
            row.push(prev);
        }

        matrix.push(row);
    }

    matrix
}

pub fn binary_super_inv_matrix(path: &Path, gf: &GF) -> GFMatrix {
    let mut matrix = Vec::new();
    let file = File::open(path).expect(
        "Binary super-invertible matrix should be created using scripts/gen_binary_supmat.py.",
    );
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let mut row = Vec::new();
        let line = line.unwrap();

        for val in line.split(" ") {
            match val {
                "0" => row.push(gf.zero()),
                "1" => row.push(gf.one()),
                _ => panic!("Binary super invertible matrix should only have binary entries"),
            }
        }

        matrix.push(row)
    }

    matrix
}

pub fn rs_gen_mat(mssg_len: usize, code_len: usize, gf: &GF) -> GFMatrix {
    debug_assert!(mssg_len <= code_len);

    let mut matrix = Vec::with_capacity(code_len);
    matrix.push(vec![gf.one(); mssg_len]);

    for i in 2..(code_len + 1) {
        let val = gf.get(i.try_into().unwrap());
        let mut row = Vec::with_capacity(mssg_len);
        row.push(gf.one());
        let mut prev = gf.one();

        for _ in 1..mssg_len {
            prev *= val;
            row.push(prev);
        }

        matrix.push(row);
    }

    matrix
}

#[derive(Clone)]
pub struct Combination(Vec<usize>);

impl Combination {
    pub fn new(map: Vec<usize>) -> Self {
        Self(map)
    }

    pub fn from_instance<T: Hash + Eq>(inp: &[T], out: &[T]) -> Self {
        let mut lookup = HashMap::new();
        for (i, v) in inp.iter().enumerate() {
            lookup.insert(v, i);
        }

        Self(out.iter().map(|v| *lookup.get(v).unwrap()).collect())
    }

    pub fn apply<T: Copy>(&self, v: &[T]) -> Vec<T> {
        self.0.iter().map(|&i| v[i]).collect()
    }
}
